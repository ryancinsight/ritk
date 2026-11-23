// Execute in the workstation page's module realm. Firefox isolates modules
// imported by WebDriver from the modules imported by the application itself.
const url = new URL("./consumer/ritk_snap.js", document.baseURI);
const {default: init} = await import(url);
const instance = await init();
Object.defineProperty(window, "studyWasmBytes", {get: () => instance.memory.buffer.byteLength});
