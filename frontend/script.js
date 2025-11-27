let selectedFile = null;

const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const resultSection = document.getElementById("result");
const predictionText = document.getElementById("predictionText");

fileInput.addEventListener("change", (e) => {
    selectedFile = e.target.files[0];

    previewImage.src = URL.createObjectURL(selectedFile);
    previewImage.classList.remove("hidden");

    predictBtn.classList.remove("hidden");
});

predictBtn.addEventListener("click", () => {
    if (!selectedFile) {
        alert("Upload an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            predictionText.textContent =
                `Tumor Type: ${data.predicted_class} | Confidence: ${data.confidence.toFixed(2)}%`;

            resultSection.classList.remove("hidden");
        })
        .catch(err => {
            console.error(err);
            alert("Backend Error!");
        });
});
