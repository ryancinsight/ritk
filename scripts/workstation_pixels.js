// Read the presented bitmap before workstation overlays or navigation can change it.
export async function screenSamples() {
  const canvas = document.querySelector("#workstation");
  // Raster exposes its displayed pixels directly. WebGPU requires a snapshot of
  // the presented bitmap because a canvas cannot switch context types.
  let context = canvas.getContext("2d");
  if (!context) {
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
    const image = await createImageBitmap(blob);
    const capture = document.createElement("canvas");
    capture.width = canvas.width;
    capture.height = canvas.height;
    context = capture.getContext("2d");
    context.drawImage(image, 0, 0);
    image.close();
  }
  const state = window.asterSlice.state();
  return state.images.map(([left, top, width, height], axis) => {
    const [x, y, w, h] = state.panes[axis];
    const [sw, sh] = [[512,512], [512,94], [512,94]][axis];
    const pixels = [];
    for (const fx of [.32,.38,.62,.69]) for (const fy of [.31,.36,.61,.68]) {
      const px = Math.floor(left + width * fx), py = Math.floor(top + height * fy);
      if (px < x || px >= x+w || py < y || py >= y+h) continue;
      pixels.push([Math.floor((px+.5-left)/(width/sw)), Math.floor((py+.5-top)/(height/sh)),
                   Array.from(context.getImageData(px,py,1,1).data)]);
    }
    return pixels;
  });
}

export async function comparePlane(name, samples) {
  const canvas = document.getElementById("ritk-snap-" + name);
  const image = canvas.getContext("2d").getImageData(0,0,canvas.width,canvas.height).data;
  const equal = samples.length > 0 && samples.every(([x,y,rgba]) =>
    rgba.every((v,i) => v === image[(y*canvas.width+x)*4+i]));
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256",image));
  return [equal, Array.from(hash, v => v.toString(16).padStart(2,"0")).join("")];
}
