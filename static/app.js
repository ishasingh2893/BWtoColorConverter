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
const guide = JSON.parse($("#crop-calibration").textContent);
const cx = guide.target_face_center_x * 240;
const cy = guide.target_face_center_y * 240;
const radius = guide.target_face_width_ratio * 120;
for (const [key, value] of Object.entries({ cx, cy, rx: radius, ry: radius })) {
  $("#face-guide").setAttribute(key, value);
}
$("#center-guide").setAttribute("d", `M${cx - 5} ${cy}h10 M${cx} ${cy - 5}v10`);
let image = null;
let scale = 1;
let x = 0;
let y = 0;
let drag = null;
let version = 0;
let loading = false;

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
  target.setTransform(size / 240, 0, 0, size / 240, 0, 0);
  target.fillStyle = "white";
  target.fillRect(0, 0, 240, 240);
  if (image) target.drawImage(image, x, y, image.width * scale, image.height * scale);
}
function changed() {
  draw();
  clearResults();
  badge.textContent = "Ready to colorize";
  status("Position the face inside the guide, then colorize.");
}
function reset() {
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
  box.focus();
  box.setPointerCapture(event.pointerId);
  drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
});
box.addEventListener("pointermove", (event) => {
  if (!drag || event.pointerId !== drag.id) return;
  const ratio = 240 / box.getBoundingClientRect().width;
  x += (event.clientX - drag.x) * ratio;
  y += (event.clientY - drag.y) * ratio;
  drag.x = event.clientX;
  drag.y = event.clientY;
  changed();
});
for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
  box.addEventListener(name, () => { drag = null; });
}
box.addEventListener("keydown", (event) => {
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
$("#reset-crop").addEventListener("click", () => { if (image) reset(); });
$("#show-guide").addEventListener("change", (event) => {
  $("#crop-guide").hidden = !event.target.checked;
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
