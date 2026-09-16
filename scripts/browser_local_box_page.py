"""Browser-page fixtures for the RITK local content-box regression."""

STYLED_CSS = """
.gallery-views {
  transform-origin: 0 0;
  transform: translate(1580px, 9.5px) scale(-0.96, 1.04);
}
.gallery-views figure:first-child {
  position: relative;
  z-index: 3;
  transform-origin: 0 0;
  transform: translate(29.5px, 17.25px) rotate(6deg) scale(1.17, 0.83);
}
.gallery-views #ritk-snap-axial {
  box-sizing: content-box !important;
  width: 641.5px !important;
  height: 641.5px !important;
  padding: 9.25px 13.5px 11.75px 17.25px !important;
  border: 6.5px solid rgb(19, 37, 53) !important;
}
"""

BORDER_BOX_CSS = """
.gallery-views {
  transform-origin: 0 0;
  scale: -0.96 1.04;
  rotate: 2deg;
  transform: translate(-1645px, 9.5px);
}
.gallery-views figure:first-child {
  position: relative;
  z-index: 3;
  transform-origin: 0 0;
  transform: translate(29.5px, 17.25px) rotate(6deg) scale(1.17, 0.83);
}
.gallery-views #ritk-snap-axial {
  box-sizing: border-box !important;
  width: 684.25px !important;
  height: 674.5px !important;
  padding: 9.25px 13.5px 11.75px 17.25px !important;
  border: 6.5px solid rgb(19, 37, 53) !important;
}
"""

GEOMETRY_SCRIPT = """
const id = arguments[0];
const fraction = arguments[1];
const canvas = document.getElementById(id);
if (!(canvas instanceof HTMLCanvasElement)) return {error: "canvas missing"};
const figure = canvas.closest("figure");
const gallery = canvas.closest(".gallery-views");
if (!(figure instanceof HTMLElement) || !(gallery instanceof HTMLElement))
  return {error: "transform ancestors missing"};

const number = (value) => {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) throw new Error(`non-finite CSS length ${value}`);
  return parsed;
};
const layoutPoint = (element) => {
  let x = 0;
  let y = 0;
  for (let current = element; current; current = current.offsetParent) {
    x += current.offsetLeft;
    y += current.offsetTop;
  }
  return {x, y};
};
const canvasStyle = getComputedStyle(canvas);
const borderLeft = number(canvasStyle.borderLeftWidth);
const borderTop = number(canvasStyle.borderTopWidth);
const borderRight = number(canvasStyle.borderRightWidth);
const borderBottom = number(canvasStyle.borderBottomWidth);
const paddingLeft = number(canvasStyle.paddingLeft);
const paddingTop = number(canvasStyle.paddingTop);
const paddingRight = number(canvasStyle.paddingRight);
const paddingBottom = number(canvasStyle.paddingBottom);
const horizontalInsets = borderLeft + borderRight + paddingLeft + paddingRight;
const verticalInsets = borderTop + borderBottom + paddingTop + paddingBottom;
const contentWidth = number(canvasStyle.width)
  - (canvasStyle.boxSizing === "border-box" ? horizontalInsets : 0);
const contentHeight = number(canvasStyle.height)
  - (canvasStyle.boxSizing === "border-box" ? verticalInsets : 0);
const borderBoxWidth = canvas.offsetWidth;
const borderBoxHeight = canvas.offsetHeight;
const measuredBorderWidth = contentWidth + horizontalInsets;
const measuredBorderHeight = contentHeight + verticalInsets;
const canvasLayout = layoutPoint(canvas);
const transforms = [figure, gallery].map((ancestor) => {
  const style = getComputedStyle(ancestor);
  const listed = style.transform === "none" ? new DOMMatrix() : new DOMMatrix(style.transform);
  const scale = style.scale === "none"
    ? new DOMMatrix()
    : new DOMMatrix().scale(...style.scale.split(" ").map(Number));
  const angle = style.rotate === "none" ? 0 : Number.parseFloat(style.rotate);
  const rotation = new DOMMatrix().rotate(angle);
  const matrix = rotation.multiply(scale).multiply(listed);
  const originParts = style.transformOrigin.split(" ");
  const origin = {x: number(originParts[0]), y: number(originParts[1])};
  const layout = layoutPoint(ancestor);
  const aroundOrigin = new DOMMatrix()
    .translate(layout.x + origin.x, layout.y + origin.y)
    .multiply(matrix)
    .translate(-(layout.x + origin.x), -(layout.y + origin.y));
  return {matrix, aroundOrigin};
});
let pageTransform = new DOMMatrix();
for (const transform of transforms) pageTransform = transform.aroundOrigin.multiply(pageTransform);
const rect = canvas.getBoundingClientRect();
const predictedCorners = [
  [0, 0],
  [measuredBorderWidth, 0],
  [0, measuredBorderHeight],
  [measuredBorderWidth, measuredBorderHeight],
].map(([x, y]) => new DOMPoint(canvasLayout.x + x, canvasLayout.y + y)
  .matrixTransform(pageTransform));
const correction = {
  x: rect.left + window.scrollX - Math.min(...predictedCorners.map(({x}) => x)),
  y: rect.top + window.scrollY - Math.min(...predictedCorners.map(({y}) => y)),
};
const frameWidth = Number(canvas.getAttribute("data-ritk-frame-width"));
const frameHeight = Number(canvas.getAttribute("data-ritk-frame-height"));
const toViewport = (borderX, borderY) => {
  const transformed = new DOMPoint(
    canvasLayout.x + borderX,
    canvasLayout.y + borderY,
  ).matrixTransform(pageTransform);
  const exact = {
    x: transformed.x + correction.x - window.scrollX,
    y: transformed.y + correction.y - window.scrollY,
  };
  const client = {x: Math.round(exact.x), y: Math.round(exact.y)};
  const target = document.elementFromPoint(client.x, client.y);
  return {
    exact,
    client,
    target_id: target && typeof target.id === "string" ? target.id : null,
  };
};
const content = toViewport(
  borderLeft + paddingLeft + contentWidth * fraction[0],
  borderTop + paddingTop + contentHeight * fraction[1],
);
const padding = paddingLeft > 0
  ? toViewport(borderLeft + paddingLeft * 0.5, borderTop + paddingTop + contentHeight * 0.5)
  : null;
const border = borderLeft > 0
  ? toViewport(borderLeft * 0.5, borderTop + paddingTop + contentHeight * 0.5)
  : null;
return {
  content,
  padding,
  border,
  content_width: contentWidth,
  content_height: contentHeight,
  box_sizing: canvasStyle.boxSizing,
  border_left: borderLeft,
  border_top: borderTop,
  padding_left: paddingLeft,
  padding_top: paddingTop,
  border_box_width: borderBoxWidth,
  border_box_height: borderBoxHeight,
  bounding_width: rect.width,
  bounding_height: rect.height,
  frame_width: frameWidth,
  frame_height: frameHeight,
  transform: Object.fromEntries(
    ["a", "b", "c", "d", "e", "f"].map((name) => [name, transforms[0].matrix[name]])
  ),
  transform_chain: transforms.map(({matrix}) => Object.fromEntries(
    ["a", "b", "c", "d", "e", "f"].map((name) => [name, matrix[name]])
  )),
};
"""

INSTALL_EVENTS_SCRIPT = """
const canvas = document.getElementById(arguments[0]);
if (!(canvas instanceof HTMLCanvasElement)) return false;
window.__ritkLocalBoxEvents = [];
for (const type of ["pointerdown", "pointerup", "wheel"]) {
  canvas.addEventListener(type, (event) => {
    window.__ritkLocalBoxEvents.push({
      type: event.type,
      is_trusted: event.isTrusted === true,
      target_id: event.target && event.target.id,
      client_x: event.clientX,
      client_y: event.clientY,
      delta_y: event.deltaY,
    });
  }, {capture: true, passive: true});
}
return true;
"""

READ_EVENTS_SCRIPT = """
const events = Array.isArray(window.__ritkLocalBoxEvents)
  ? window.__ritkLocalBoxEvents.splice(0)
  : null;
return events;
"""

SLICE_VECTOR_SCRIPT = """
return arguments[0].map((id) => {
  const canvas = document.getElementById(id);
  return {
    index: Number(canvas.getAttribute("data-ritk-slice-index")),
    count: Number(canvas.getAttribute("data-ritk-slice-count")),
  };
});
"""

FRAME_GENERATION_SCRIPT = """
return Number(document.getElementById(arguments[0]).getAttribute("data-ritk-frame-generation"));
"""

WAIT_FOR_FRAME_SCRIPT = """
const id = arguments[0];
const previous = arguments[1];
const done = arguments[arguments.length - 1];
let remaining = 120;
const check = () => {
  const canvas = document.getElementById(id);
  const current = Number(canvas && canvas.getAttribute("data-ritk-frame-generation"));
  if (Number.isSafeInteger(current) && current > previous) {
    done({ok: true, generation: current});
    return;
  }
  if (remaining === 0) {
    done({ok: false, generation: current});
    return;
  }
  remaining -= 1;
  requestAnimationFrame(check);
};
requestAnimationFrame(check);
"""
