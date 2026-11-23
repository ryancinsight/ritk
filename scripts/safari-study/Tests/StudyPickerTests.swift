import XCTest

final class StudyPickerTests: XCTestCase {
    func testPublicStudyPicker() throws {
        continueAfterFailure = false
        let safari = XCUIApplication(bundleIdentifier: "com.apple.Safari")
        safari.activate()
        safari.typeKey("l", modifierFlags: .command)
        safari.typeText("http://127.0.0.1:17823/")
        safari.typeKey(.return, modifierFlags: [])
        let choose = safari.buttons["Choose study files"]
        XCTAssertTrue(choose.waitForExistence(timeout: 20), safari.debugDescription)
        choose.click()
        let open = safari.sheets["open-panel"].buttons["Upload"].firstMatch
        XCTAssertTrue(open.waitForExistence(timeout: 10), safari.debugDescription)
        safari.typeKey("g", modifierFlags: [.command, .shift])
        safari.typeText("/Users/Shared/RITKPublicStudy/00000001.dcm")
        safari.typeKey(.return, modifierFlags: [])
        let first = safari.descendants(matching: .any)
            .matching(NSPredicate(format: "label == %@ OR value == %@ OR title == %@",
                                  "00000001.dcm", "00000001.dcm", "00000001.dcm")).firstMatch
        XCTAssertTrue(first.waitForExistence(timeout: 12), safari.sheets.debugDescription)
        first.click()
        safari.typeKey("a", modifierFlags: .command)
        open.click()
        XCTAssertTrue(safari.staticTexts["All 94 file digests match"].waitForExistence(timeout: 30),
                      safari.debugDescription)
    }
}
