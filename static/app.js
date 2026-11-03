const $ = (selector) => document.querySelector(selector);
const form = $("#colorize-form");
const photoInput = $("#photo-input");
const urlInput = $("#image-url");
const canvas = $("#crop-canvas");
const context = canvas.getContext("2d");
const box = $("#crop-box");
const zoom = $("#crop-zoom");
const submit = $("#submit-button");
const badge = $("#result-badge");
let image = null;
let scale = 1;
let x = 0;
let y = 0;
let drag = null;
let version = 0;
let loading = false;
const guideData = JSON.parse($("#portrait-guide-data").textContent);
const outlineHandles = [$("#neck-left"), $("#neck-right"), $("#shoulder-left"), $("#shoulder-right"), $("#head-left"), $("#head-right"), $("#cheek-left"), $("#cheek-right"), $("#crown-top")];
let outlineOffsets = guideData.anchors.map(() => ({ x: 0, y: 0 }));
const cropSurface = document.createElement("canvas");
cropSurface.width = cropSurface.height = 240;
const cropContext = cropSurface.getContext("2d", { willReadFrequently: true });
function updateOutlineHandles() {
  outlineHandles.forEach((handle, i) => {
    handle.style.left = `${(guideData.anchors[i].x + outlineOffsets[i].x) / 2.4}%`;
    handle.style.top = `${(guideData.anchors[i].y + outlineOffsets[i].y) / 2.4}%`;
  });
  const outline = PortraitBoundary.path(PortraitBoundary.contour(guideData.points, guideData.anchors, outlineOffsets));
  $("#guide-shade").setAttribute("d", `M0 0H240V240H0Z ${outline} L240 240H0Z`);
  $("#guide-outline-shadow").setAttribute("d", outline);
  $("#guide-outline").setAttribute("d", outline);
}
function resetOutline() {
  outlineOffsets = guideData.anchors.map(() => ({ x: 0, y: 0 }));
  updateOutlineHandles();
}
updateOutlineHandles();

function status(message, error = false) {
  $("#status").textContent = message;
  $("#status").classList.toggle("error", error);
}
function showImage(id, emptyId, src) {
  $(id).src = src;
  $(id).hidden = false;
  $(emptyId).hidden = true;
}
function clearResults() {
  for (const name of ["processed-input", "linear-output", "knn-output", "hybrid-output"]) {
    $(`#${name}-preview`).hidden = true;
  }
  for (const id of ["processed-input", "linear-result", "knn-result", "hybrid-result"]) {
    $(`#empty-${id}`).hidden = false;
  }
}
function draw(target = context, size = 480) {
  cropContext.setTransform(1, 0, 0, 1, 0, 0);
  cropContext.fillStyle = "white";
  cropContext.fillRect(0, 0, 240, 240);
  if (image) cropContext.drawImage(image, x, y, image.width * scale, image.height * scale);
  target.setTransform(1, 0, 0, 1, 0, 0);
  target.clearRect(0, 0, size, size);
  target.drawImage(cropSurface, 0, 0, size, size);
}
function changed() {
  draw();
  clearResults();
  badge.textContent = "Ready to colorize";
  status("Position the portrait, then drag the outline points to fit the outline around it.");
}
function reset() {
  resetOutline();
  zoom.value = "1";
  $("#zoom-value").textContent = "100%";
  scale = Math.min(240 / image.width, 240 / image.height);
  x = (240 - image.width * scale) / 2;
  y = (240 - image.height * scale) / 2;
  changed();
}
function invalidateSource() {
  version++;
  image = null;
  drag = null;
  submit.disabled = true;
  $("#crop-editor").hidden = true;
  $("#crop-placeholder").hidden = false;
  $("#raw-input-preview").hidden = true;
  $("#empty-raw-input").hidden = false;
  clearResults();
  badge.textContent = "Choose a portrait";
}
async function prepare(file) {
  invalidateSource();
  const current = version;
  const body = new FormData();
  if (file) {
    if (file.size > 12 * 1024 * 1024) {
      status("Image is too large. Use a file under 12 MB.", true);
      return;
    }
    body.append("photo", file);
  } else {
    if (!urlInput.value.trim() || !urlInput.checkValidity()) {
      status("Enter a valid photo link first.", true);
      return;
    }
    body.append("image_url", urlInput.value.trim());
  }
  status("Loading portrait…");
  try {
    const response = await fetch("/api/prepare", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Unable to load this portrait.");
    const loaded = new Image();
    loaded.src = data.image;
    await loaded.decode();
    if (current !== version) return;
    image = loaded;
    showImage("#raw-input-preview", "#empty-raw-input", data.image);
    $("#crop-editor").hidden = false;
    $("#crop-placeholder").hidden = true;
    submit.disabled = false;
    reset();
  } catch (error) {
    if (current === version) status(error.message, true);
  }
}
photoInput.addEventListener("change", () => {
  if (photoInput.files[0]) {
    urlInput.value = "";
    prepare(photoInput.files[0]);
  }
});
$("#load-link").addEventListener("click", () => { photoInput.value = ""; prepare(); });
urlInput.addEventListener("input", () => {
  photoInput.value = "";
  invalidateSource();
  status("Click Load photo link to position this portrait.");
});
urlInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") { event.preventDefault(); $("#load-link").click(); }
});
const dropzone = $("#dropzone");
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  if (!loading) dropzone.classList.add("drag");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("drag");
  if (loading) return;
  const file = event.dataTransfer.files[0];
  if (file) { photoInput.value = ""; urlInput.value = ""; prepare(file); }
});
box.addEventListener("pointerdown", (event) => {
  if (!image || loading || (event.pointerType === "mouse" && event.button !== 0)) return;
  event.preventDefault();
  const neck = outlineHandles.indexOf(event.target);
  if (neck >= 0) outlineHandles[neck].focus();
  else box.focus();
  box.setPointerCapture(event.pointerId);
  drag = { id: event.pointerId, x: event.clientX, y: event.clientY, neck };
});
box.addEventListener("pointermove", (event) => {
  if (!drag || event.pointerId !== drag.id) return;
  const ratio = 240 / box.getBoundingClientRect().width;
  const dx = (event.clientX - drag.x) * ratio;
  const dy = (event.clientY - drag.y) * ratio;
  if (drag.neck >= 0) {
    moveOutline(drag.neck, dx, dy);
  } else {
    x += dx;
    y += dy;
  }
  drag.x = event.clientX;
  drag.y = event.clientY;
  changed();
});
for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
  box.addEventListener(name, () => { drag = null; });
}
box.addEventListener("keydown", (event) => {
  if (outlineHandles.includes(event.target)) return;
  if (!image || loading) return;
  const delta = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
  if (!delta) return;
  event.preventDefault();
  const step = event.shiftKey ? 10 : 1;
  x += delta[0] * step;
  y += delta[1] * step;
  changed();
});
zoom.addEventListener("input", () => {
  if (!image) return;
  const next = Math.min(240 / image.width, 240 / image.height) * Number(zoom.value);
  const ratio = next / scale;
  x = 120 - (120 - x) * ratio;
  y = 120 - (120 - y) * ratio;
  scale = next;
  $("#zoom-value").textContent = `${Math.round(Number(zoom.value) * 100)}%`;
  changed();
});
function moveOutline(index, dx, dy) {
  outlineOffsets[index] = index === 2 || index === 3
    ? PortraitBoundary.edgeOffset(index, dx, dy, outlineOffsets, guideData.anchors)
    : PortraitBoundary.controlOffset(index, dx, dy, outlineOffsets, guideData.anchors,
      Math.min(...guideData.points.map((point) => point.y)));
  updateOutlineHandles();
}
outlineHandles.forEach((handle, index) => {
  handle.addEventListener("keydown", (event) => {
    if (!image || loading) return;
    const delta = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
    if (!delta) return;
    event.preventDefault();
    event.stopPropagation();
    const step = event.shiftKey ? 5 : 1;
    moveOutline(index, delta[0] * step, delta[1] * step);
    changed();
  });
});
$("#reset-neck").addEventListener("click", () => {
  if (image && !loading) { resetOutline(); changed(); }
});
$("#reset-crop").addEventListener("click", () => { if (image) reset(); });
$("#show-guide").addEventListener("change", (event) => {
  $("#crop-guide").hidden = !event.target.checked;
  outlineHandles.forEach((handle) => { handle.hidden = !event.target.checked; });
});
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!image || loading) return;
  loading = true;
  drag = null;
  submit.disabled = true;
  $("#input-controls").disabled = true;
  badge.textContent = "Working";
  status("Colorizing your crop…");
  try {
    const output = document.createElement("canvas");
    output.width = output.height = 240;
    draw(output.getContext("2d"), 240);
    const blob = await new Promise((resolve) => output.toBlob(resolve, "image/png"));
    if (!blob) throw new Error("Unable to create crop. Please try again.");
    const body = new FormData();
    body.append("photo", blob, "portrait-crop.png");
    body.append("manual_crop", "true");
    const response = await fetch("/api/colorize", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Unable to colorize this portrait.");
    showImage("#processed-input-preview", "#empty-processed-input", data.input);
    for (const name of ["linear", "knn", "hybrid"]) {
      if (data.outputs[name]) {
        showImage(`#${name}-output-preview`, `#empty-${name}-result`, data.outputs[name]);
      } else {
        $(`#empty-${name}-result strong`).textContent = `${name.toUpperCase()} model unavailable.`;
      }
    }
    badge.textContent = "Complete";
    status("Model comparison ready. Adjust the crop to try another framing.");
  } catch (error) {
    badge.textContent = "Needs attention";
    status(error.message, true);
  } finally {
    loading = false;
    submit.disabled = !image;
    $("#input-controls").disabled = false;
  }
});
