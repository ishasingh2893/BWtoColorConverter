const form = document.querySelector("#colorize-form");
const dropzone = document.querySelector("#dropzone");
const photoInput = document.querySelector("#photo-input");
const imageUrlInput = document.querySelector("#image-url");
const emptyRawInput = document.querySelector("#empty-raw-input");
const rawInputPreview = document.querySelector("#raw-input-preview");
const emptyProcessedInput = document.querySelector("#empty-processed-input");
const processedInputPreview = document.querySelector("#processed-input-preview");
const maskPreviews = {
  rembg: {
    image: document.querySelector("#rembg-subject-mask-preview"),
    empty: document.querySelector("#empty-rembg-subject-mask"),
  },
  sobel_rembg: {
    image: document.querySelector("#sobel-rembg-subject-mask-preview"),
    empty: document.querySelector("#empty-sobel-rembg-subject-mask"),
  },
  mediapipe: {
    image: document.querySelector("#mediapipe-subject-mask-preview"),
    empty: document.querySelector("#empty-mediapipe-subject-mask"),
  },
  geometry: {
    image: document.querySelector("#geometry-subject-mask-preview"),
    empty: document.querySelector("#empty-geometry-subject-mask"),
  },
};
const modelPreviews = {
  linear_rembg: {
    image: document.querySelector("#linear-rembg-output-preview"),
    empty: document.querySelector("#empty-linear-rembg-result"),
    missing: "Linear model unavailable.",
  },
  linear_sobel_rembg: {
    image: document.querySelector("#linear-sobel-rembg-output-preview"),
    empty: document.querySelector("#empty-linear-sobel-rembg-result"),
    missing: "Linear model unavailable.",
  },
  linear_mediapipe: {
    image: document.querySelector("#linear-mediapipe-output-preview"),
    empty: document.querySelector("#empty-linear-mediapipe-result"),
    missing: "Linear model unavailable.",
  },
  linear_geometry: {
    image: document.querySelector("#linear-geometry-output-preview"),
    empty: document.querySelector("#empty-linear-geometry-result"),
    missing: "Linear model unavailable.",
  },
  knn_rembg: {
    image: document.querySelector("#knn-rembg-output-preview"),
    empty: document.querySelector("#empty-knn-rembg-result"),
    missing: "KNN model unavailable.",
  },
  knn_sobel_rembg: {
    image: document.querySelector("#knn-sobel-rembg-output-preview"),
    empty: document.querySelector("#empty-knn-sobel-rembg-result"),
    missing: "KNN model unavailable.",
  },
  knn_mediapipe: {
    image: document.querySelector("#knn-mediapipe-output-preview"),
    empty: document.querySelector("#empty-knn-mediapipe-result"),
    missing: "KNN model unavailable.",
  },
  knn_geometry: {
    image: document.querySelector("#knn-geometry-output-preview"),
    empty: document.querySelector("#empty-knn-geometry-result"),
    missing: "KNN model unavailable.",
  },
  hybrid_rembg: {
    image: document.querySelector("#hybrid-rembg-output-preview"),
    empty: document.querySelector("#empty-hybrid-rembg-result"),
    missing: "Hybrid model unavailable.",
  },
  hybrid_sobel_rembg: {
    image: document.querySelector("#hybrid-sobel-rembg-output-preview"),
    empty: document.querySelector("#empty-hybrid-sobel-rembg-result"),
    missing: "Hybrid model unavailable.",
  },
  hybrid_mediapipe: {
    image: document.querySelector("#hybrid-mediapipe-output-preview"),
    empty: document.querySelector("#empty-hybrid-mediapipe-result"),
    missing: "Hybrid model unavailable.",
  },
  hybrid_geometry: {
    image: document.querySelector("#hybrid-geometry-output-preview"),
    empty: document.querySelector("#empty-hybrid-geometry-result"),
    missing: "Hybrid model unavailable.",
  },
};
const statusEl = document.querySelector("#status");
const resultBadge = document.querySelector("#result-badge");
const submitButton = document.querySelector("#submit-button");

Object.values(modelPreviews).forEach((preview) => {
  preview.defaultTitle = preview.empty.querySelector("strong")?.textContent || "";
});

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function resetModelPreviews() {
  Object.values(maskPreviews).forEach(({ image, empty }) => {
    image.hidden = true;
    empty.hidden = false;
  });
  Object.values(modelPreviews).forEach((preview) => {
    preview.image.hidden = true;
    preview.empty.hidden = false;
    preview.empty.classList.remove("unavailable");
    const strong = preview.empty.querySelector("strong");
    if (strong) strong.textContent = preview.defaultTitle;
  });
}

function setUnavailablePreviews() {
  Object.values(modelPreviews).forEach(({ image, empty, missing }) => {
    image.hidden = true;
    empty.hidden = false;
    empty.classList.add("unavailable");
    const strong = empty.querySelector("strong");
    if (strong) strong.textContent = missing;
  });
}

function previewFile(file) {
  if (!file) return;
  rawInputPreview.src = URL.createObjectURL(file);
  rawInputPreview.hidden = false;
  emptyRawInput.hidden = true;
  processedInputPreview.hidden = true;
  emptyProcessedInput.hidden = false;
  resetModelPreviews();
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
  rawInputPreview.src = imageUrlInput.value;
  rawInputPreview.hidden = false;
  emptyRawInput.hidden = true;
  processedInputPreview.hidden = true;
  emptyProcessedInput.hidden = false;
  resetModelPreviews();
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

    if (data.input) {
      processedInputPreview.src = data.input;
      processedInputPreview.hidden = false;
      emptyProcessedInput.hidden = true;
    }
    Object.entries(data.subject_masks || {}).forEach(([key, value]) => {
      const preview = maskPreviews[key];
      if (!preview || !value) return;
      preview.image.src = value;
      preview.image.hidden = false;
      preview.empty.hidden = true;
    });

    setUnavailablePreviews();
    Object.entries(data.outputs || {}).forEach(([key, value]) => {
      const preview = modelPreviews[key];
      if (!preview || !value) return;
      preview.image.src = value;
      preview.image.hidden = false;
      preview.empty.hidden = true;
      preview.empty.classList.remove("unavailable");
    });
    resultBadge.textContent = "Complete";
    setStatus("Model comparison ready.");
  } catch (error) {
    resultBadge.textContent = "Needs attention";
    setStatus(error.message, true);
  } finally {
    submitButton.disabled = false;
  }
});
