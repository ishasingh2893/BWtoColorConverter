const form = document.querySelector("#colorize-form");
const dropzone = document.querySelector("#dropzone");
const photoInput = document.querySelector("#photo-input");
const imageUrlInput = document.querySelector("#image-url");
const emptyInput = document.querySelector("#empty-input");
const inputPreview = document.querySelector("#input-preview");
const outputPreview = document.querySelector("#output-preview");
const emptyResult = document.querySelector("#empty-result");
const statusEl = document.querySelector("#status");
const resultBadge = document.querySelector("#result-badge");
const submitButton = document.querySelector("#submit-button");

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function previewFile(file) {
  if (!file) return;
  inputPreview.src = URL.createObjectURL(file);
  inputPreview.hidden = false;
  emptyInput.hidden = true;
  imageUrlInput.value = "";
  resultBadge.textContent = "Ready";
}

photoInput.addEventListener("change", () => {
  previewFile(photoInput.files[0]);
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("drag");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("drag");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("drag");
  const file = event.dataTransfer.files[0];
  if (!file) return;
  photoInput.files = event.dataTransfer.files;
  previewFile(file);
});

imageUrlInput.addEventListener("input", () => {
  if (!imageUrlInput.value) return;
  photoInput.value = "";
  inputPreview.src = imageUrlInput.value;
  inputPreview.hidden = false;
  emptyInput.hidden = true;
  resultBadge.textContent = "Linked";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = new FormData(form);

  if (!photoInput.files[0] && !imageUrlInput.value.trim()) {
    setStatus("Upload a portrait or paste a photo link.", true);
    return;
  }

  submitButton.disabled = true;
  resultBadge.textContent = "Working";
  setStatus("Colorizing portrait...");

  try {
    const response = await fetch("/api/colorize", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Unable to colorize this portrait.");

    outputPreview.src = data.output;
    outputPreview.hidden = false;
    emptyResult.hidden = true;
    resultBadge.textContent = "Complete";
    setStatus("Colorized portrait ready.");
  } catch (error) {
    resultBadge.textContent = "Needs attention";
    setStatus(error.message, true);
  } finally {
    submitButton.disabled = false;
  }
});
