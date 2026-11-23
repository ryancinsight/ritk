import XCTest

final class StudyPickerTests: XCTestCase {
    let safari = XCUIApplication(bundleIdentifier: "com.apple.Safari")
    typealias State = [String: Any]

    @discardableResult
    func wait(_ description: String, _ test: (State) -> Bool,
              file: StaticString = #filePath, line: UInt = #line) -> State {
        let deadline = Date().addingTimeInterval(30)
        var last = ""
        repeat {
            let view = safari.images.matching(NSPredicate(format: "label BEGINSWITH %@", "RITK {")).firstMatch
            if view.exists {
                last = view.label
                if last.hasPrefix("RITK "),
                   let state = try? JSONSerialization.jsonObject(with: Data(last.dropFirst(5).utf8)) as? State,
                   state["error"] != nil || test(state) {
                    XCTAssertNil(state["error"], description + ": " + last, file: file, line: line)
                    return state
                }
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.15))
        } while Date() < deadline
        XCTFail("\(description). Last state: \(last)", file: file, line: line)
        return [:]
    }

    func choose(_ directory: String, embedded: Bool = false) {
        if embedded { safari.buttons["Open reference study"].click() }
        else { point(62,53).click() }  // Open Study in the workstation toolbar.
        let panel = safari.sheets["open-panel"]
        let upload = panel.buttons["Upload"].firstMatch
        if !upload.waitForExistence(timeout: 10) {
            let screen = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
            screen.name = "Open Study"
            screen.lifetime = .keepAlways
            add(screen)
            XCTFail("Open Study panel is unavailable: " + safari.debugDescription)
        }
        safari.typeKey("2", modifierFlags: .command)  // Standard macOS list view.
        let filename = directory + "-00000001.dcm"
        safari.typeKey("g", modifierFlags: [.command, .shift])
        safari.typeText("/Users/Shared/RITKPublicStudy/" + directory + "/" + filename)
        safari.typeKey(.return, modifierFlags: [])
        let list = panel.outlines["ListView"]
        let first = list.textFields.matching(NSPredicate(format: "value == %@", filename)).firstMatch
        XCTAssertTrue(first.waitForExistence(timeout: 12), panel.debugDescription)
        // Click the visible file in panel coordinates. Asking XCTest to scroll an
        // AppKit filename editor into view can invalidate its accessibility row.
        let file = first.frame, sheet = panel.frame
        let center = CGPoint(x: file.midX, y: file.midY)
        XCTAssertTrue(sheet.contains(center), "The selected filename must be visible")
        panel.coordinate(withNormalizedOffset: .zero)
            .withOffset(CGVector(dx: center.x - sheet.minX, dy: center.y - sheet.minY)).click()
        safari.typeKey("a", modifierFlags: .command)
        if directory != "File" {
            let second = list.textFields.matching(NSPredicate(format: "value == %@",
                directory + "-00000002.dcm")).firstMatch
            XCTAssertTrue(second.isSelected, "The native list must select multiple files before upload")
        }
        upload.click()
    }

    func point(_ x: Int, _ y: Int) -> XCUICoordinate {
        safari.webViews.firstMatch.coordinate(withNormalizedOffset: .zero)
            .withOffset(CGVector(dx: x, dy: y))
    }

    func openPositionPair(_ name: String, targetCount: Int) throws -> Int {
        choose(name)
        wait("two positioned acquisitions") { $0["series"] as? Int == 2 && ($0["panels"] as? [[Int]])?.count == 1 }
        safari.typeKey(.F4, modifierFlags: [])
        for key: XCUIKeyboardKey in [.space, .downArrow, .space, .return] { safari.typeKey(key, modifierFlags: []) }
        let opened = wait("positioned pair displayed") { state in
            (state["panels"] as? [[Int]])?.map { $0[1] }.sorted() == [5, targetCount].sorted()
        }
        let panels = try XCTUnwrap(opened["panels"] as? [[Int]])
        let source = try XCTUnwrap(panels.firstIndex { $0[1] == 5 })
        if opened["panel"] as? Int != source { safari.typeKey(.tab, modifierFlags: []) }
        return source
    }

    func testPatientPosition() throws {
        continueAfterFailure = false
        safari.open(try XCTUnwrap(URL(string: "http://127.0.0.1:17823/index.html?position")))
        wait("workstation ready") { $0["width"] as? Int ?? 0 > 0 }
        let source = try openPositionPair("Anchors", targetCount: 9)
        func positions(_ active: Int, _ other: Int) -> (State) -> Bool {
            { state in
                guard let p = state["panels"] as? [[Int]], p.count == 2 else { return false }
                return p[source][0] == active && p[1-source][0] == other
            }
        }
        safari.typeKey(.downArrow, modifierFlags: [])
        wait("select independent anatomical landmarks", positions(3, 4))
        safari.typeKey(.F5, modifierFlags: .shift)
        safari.typeKey(.upArrow, modifierFlags: [])
        wait("two millimetres from the anchors", positions(2, 2))
        safari.typeKey(.tab, modifierFlags: [])
        for _ in 0..<2 { safari.typeKey(.downArrow, modifierFlags: []) }
        wait("scroll back from the other study", positions(3, 4))

        let aligned = wait("manually aligned source pane") { $0["panel"] as? Int == 1-source }
        let images = try XCTUnwrap(aligned["images"] as? [[Double]])
        let pan = try XCTUnwrap(aligned["panes"] as? [[Int]])[1-source]
        let x = pan[0]+pan[2]/2, y = pan[1]+pan[3]/2
        safari.typeKey("m", modifierFlags: [])
        point(x,y).press(forDuration: 0.1, thenDragTo: point(x+24,y-12))
        wait("manual link transfers the pan displacement") {
            guard let moved = $0["images"] as? [[Double]] else {return false}
            let dx = moved[1-source][0]-images[1-source][0]
            let dy = moved[1-source][1]-images[1-source][1]
            return dx > 0 && dy < 0 && dx == moved[source][0]-images[source][0]
                && dy == moved[source][1]-images[source][1]
        }
        safari.typeKey(.F6, modifierFlags: .shift)
        safari.typeKey("2", modifierFlags: [])
        let preset = wait("source window preset") { $0["window"] as? [Double] != aligned["window"] as? [Double] }
        safari.typeKey(.tab, modifierFlags: [])
        wait("target window follows manual link") { $0["panel"] as? Int == source && $0["window"] as? [Double] == preset["window"] as? [Double] }

        let axial = try openPositionPair("Intersection", targetCount: 3)
        func mark(_ panel: Int, _ sample: Int, _ visible: Bool) -> (State) -> Bool {
            { state in
                guard let m = state["marks"] as? [[Bool]], m.count == 2 else { return false }
                return m[panel][sample] == visible
            }
        }
        let initial = wait("yellow reference at four millimetres", mark(1-axial, 0, true))
        let image = try XCTUnwrap(initial["images"] as? [[Double]])[axial]
        safari.typeKey("q", modifierFlags: [])
        point(Int((image[0]+0.25*image[2]).rounded()),Int((image[1]+0.75*image[3]).rounded())).press(forDuration: 1.0)
        wait("held cursor selected a sagittal plane and release restored the reference") {
            $0["heldCursorSeen"] as? Bool == true && ($0["panels"] as? [[Int]])?[1-axial][0] == 0 && mark(1-axial,0,true)($0)
        }
        safari.typeKey(.tab, modifierFlags: [])
        safari.typeKey(.downArrow, modifierFlags: [])
        safari.typeKey(.tab, modifierFlags: [])
        safari.typeKey(.downArrow, modifierFlags: [])
        wait("yellow reference at six millimetres", mark(1-axial, 1, true))
        safari.typeKey(.F12, modifierFlags: .control)
        wait("hidden reference", mark(1-axial, 1, false))
        safari.typeKey(.F12, modifierFlags: .control)
        safari.typeKey(.tab, modifierFlags: [])
        wait("reciprocal reference", mark(axial, 2, true))
        safari.typeKey("r", modifierFlags: [])
        safari.typeKey(.tab, modifierFlags: [])
        wait("rotated reference", mark(1-axial, 3, true))
    }

    func testPublicStudy() throws {
        continueAfterFailure = false
        let url = try XCTUnwrap(URL(string: "http://127.0.0.1:17823/index.html"))
        safari.open(url)
        let ready = wait("workstation layout") { $0["width"] as? Int ?? 0 > 0 }
        let panes = try XCTUnwrap(ready["panes"] as? [[Int]])
        XCTAssertEqual(Array(panes[0].prefix(2)), [276, 78])
        let x = panes[0][0] + panes[0][2]/2, y = panes[0][1] + panes[0][3]/2
        point(x, y).click()
        choose("MRI")
        let initial = wait("94 decoded MR slices and three sampled planes") {
            $0["captured"] as? Bool == true && $0["counts"] as? [Int] == [94,512,512] &&
            ($0["indices"] as? [Int])?.first == 47 && $0["selection"] as? [Int] == [94,49807236]
        }
        let screenshot = XCTAttachment(screenshot: safari.screenshot())
        screenshot.name = "Native Safari MRI workstation"
        screenshot.lifetime = .keepAlways
        add(screenshot)
        for (dir, selection) in [("Count", [513,0]), ("File", [1,(64 << 20)+1]), ("Batch", [5,320 << 20])] {
            choose(dir)
            wait("host rejection preserved the displayed MRI: " + dir) {
                $0["selection"] as? [Int] == selection && $0["rejected"] as? Bool == true
            }
        }
        point(x,y).scroll(byDeltaX: 0, deltaY: 1)
        wait("one axial step") { abs((($0["indices"] as? [Int])?.first ?? 47) - 47) == 1 }
        point(555,53).click()
        wait("ruler selected") { $0["tool"] as? String == "Length" }
        point(x-20,y).press(forDuration: 0.1, thenDragTo: point(x+20,y))
        wait("positive ruler measurement") { $0["annotations"] as? Int == 1 && $0["measurement"] as? Double ?? 0 > 0 }
        point(345,53).click()
        point(x,y).press(forDuration: 0.1, thenDragTo: point(x+18,y+20))
        wait("window and level changed") { $0["window"] as? [Double] != initial["window"] as? [Double] }
        point(480,53).click()
        point(x,y).press(forDuration: 0.1, thenDragTo: point(x,y+20))
        wait("zoom changed") { $0["zoom"] as? Double != 1 }
        safari.typeKey("r", modifierFlags: [])
        wait("rotated") { $0["orientationIdentity"] as? Bool == false }
        safari.typeKey(.space, modifierFlags: [])
        let cine = wait("cine playing") { $0["cine"] as? Bool == true }
        wait("cine advanced") { $0["indices"] as? [Int] != cine["indices"] as? [Int] }
        safari.typeKey(.space, modifierFlags: [])
        wait("cine stopped") { $0["cine"] as? Bool == false }
        choose("Comparison")
        wait("two independent series") { $0["series"] as? Int == 2 }
        safari.typeKey(.F4, modifierFlags: [])
        for key: XCUIKeyboardKey in [.space, .downArrow, .space, .return] { safari.typeKey(key, modifierFlags: []) }
        wait("both series open") { ($0["panels"] as? [[Int]])?.map { $0[1] }.sorted() == [2,94] }
        safari.typeKey(.tab, modifierFlags: [])
        let active = wait("second comparison panel active") { $0["panel"] as? Int == 1 }
        let panels = try XCTUnwrap(active["panels"] as? [[Int]])
        let mr = try XCTUnwrap(panels.firstIndex { $0[1] == 94 })
        let bounds = try XCTUnwrap(active["panes"] as? [[Int]])[mr]
        point(bounds[0]+bounds[2]/2, bounds[1]+bounds[3]/2).scroll(byDeltaX: 0, deltaY: 1)
        wait("independent scroll") { state in
            guard let actual = state["panels"] as? [[Int]] else { return false }
            return abs(actual[mr][0]-panels[mr][0]) == 1 && actual[1-mr] == panels[1-mr]
        }
        // Safari page zoom changes the real CSS viewport on the hosted desktop.
        for _ in 0..<5 { safari.typeKey("+", modifierFlags: .command) }
        wait("narrow layout fits the viewport") {
            ($0["panes"] as? [[Int]])?.first?.first == 0 && $0["overflow"] as? Bool == false
        }
        safari.typeKey("0", modifierFlags: .command)
        safari.links["Open projection entry"].click()
        wait("MIP entry") { $0["projection"] as? Bool == true }
        safari.links["Open independent reference"].click()
        XCTAssertTrue(safari.buttons["Open reference study"].waitForExistence(timeout: 20))
        choose("MRI", embedded: true)
        let result = wait("three complete reference plane digests and listener cleanup") {
            $0["planes"] as? Int == 3 && ($0["before"] as? Int ?? 0) > 0 && $0["after"] as? Int == 0
        }
        let displayed = try XCTUnwrap(result["displayed"] as? State)
        let evidence = XCTAttachment(data: try JSONSerialization.data(withJSONObject: displayed),
                                     uniformTypeIdentifier: "public.json")
        evidence.name = "Displayed MRI planes"
        evidence.lifetime = .keepAlways
        add(evidence)
    }
}
