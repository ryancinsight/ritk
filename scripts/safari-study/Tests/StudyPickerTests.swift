import XCTest

final class StudyPickerTests: XCTestCase {
    let safari = XCUIApplication(bundleIdentifier: "com.apple.Safari")
    typealias State = [String: Any]

    func go(_ path: String) {
        safari.typeKey("l", modifierFlags: .command)
        safari.typeText("http://127.0.0.1:17823/" + path)
        safari.typeKey(.return, modifierFlags: [])
    }

    @discardableResult
    func wait(_ description: String, _ test: (State) -> Bool,
              file: StaticString = #filePath, line: UInt = #line) -> State {
        let deadline = Date().addingTimeInterval(30)
        var last = ""
        repeat {
            let view = safari.webViews.firstMatch
            if view.exists {
                last = view.label
                if last.hasPrefix("RITK "),
                   let state = try? JSONSerialization.jsonObject(with: Data(last.dropFirst(5).utf8)) as? State,
                   test(state) { return state }
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.15))
        } while Date() < deadline
        XCTFail("\(description). Last state: \(last)", file: file, line: line)
        return [:]
    }

    func choose(_ directory: String, embedded: Bool = false) {
        if embedded { safari.buttons["Open reference study"].click() }
        else { safari.typeKey("o", modifierFlags: .control) }
        let panel = safari.sheets["open-panel"]
        let upload = panel.buttons["Upload"].firstMatch
        XCTAssertTrue(upload.waitForExistence(timeout: 10), safari.debugDescription)
        safari.typeKey("g", modifierFlags: [.command, .shift])
        safari.typeText("/Users/Shared/RITKPublicStudy/" + directory + "/00000001.dcm")
        safari.typeKey(.return, modifierFlags: [])
        let first = panel.descendants(matching: .any)
            .matching(NSPredicate(format: "label == %@ OR value == %@ OR title == %@",
                                  "00000001.dcm", "00000001.dcm", "00000001.dcm")).firstMatch
        XCTAssertTrue(first.waitForExistence(timeout: 12), panel.debugDescription)
        first.click()
        safari.typeKey("a", modifierFlags: .command)
        upload.click()
    }

    func point(_ x: Int, _ y: Int) -> XCUICoordinate {
        safari.webViews.firstMatch.coordinate(withNormalizedOffset: .zero)
            .withOffset(CGVector(dx: x, dy: y))
    }

    func testPublicStudy() throws {
        continueAfterFailure = false
        safari.activate()
        go("")
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
        go("gallery.html?projection=mip")
        wait("MIP entry") { $0["projection"] as? Bool == true }
        go("embedded/embed.html")
        XCTAssertTrue(safari.buttons["Open reference study"].waitForExistence(timeout: 20))
        choose("MRI", embedded: true)
        wait("all three full-plane digests, screen samples and listener cleanup") {
            $0["planes"] as? Int == 3 && ($0["before"] as? Int ?? 0) > 0 && $0["after"] as? Int == 0
        }
    }
}
