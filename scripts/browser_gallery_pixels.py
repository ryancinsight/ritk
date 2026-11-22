"""Read native-resolution RGBA samples from a presented anatomical canvas."""

READ_CANVAS_PIXELS = """
const samplePixels = async (canvas) => {
  let context = canvas.getContext("2d", {willReadFrequently: true});
  let source = "canvas-2d";
  if (!context) {
    if (!canvas.getContext("webgpu")) throw new Error("unsupported canvas context");
    // Serialize the presented bitmap before the GPU surface is recycled.
    // Copying the live canvas into a second context can read a cleared frame.
    const png = await new Promise((resolve, reject) => canvas.toBlob(
      blob => blob ? resolve(blob) : reject(new Error("could not serialize WebGPU frame")),
      "image/png"));
    const image = await createImageBitmap(png);
    try {
      const target = new OffscreenCanvas(image.width, image.height);
      context = target.getContext("2d", {willReadFrequently: true});
      context.drawImage(image, 0, 0);
    } finally {
      image.close();
    }
    source = "webgpu-bitmap";
  }
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const digest = await crypto.subtle.digest("SHA-256", pixels);
  return {pixel_source: source, rgba_sha256: Array.from(new Uint8Array(digest),
    byte => byte.toString(16).padStart(2, "0")).join("")};
};
"""
