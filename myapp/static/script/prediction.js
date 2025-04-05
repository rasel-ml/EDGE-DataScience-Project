function togglePrediction() {
    let predictionType = document.getElementById("predictionType").value;
    let temperatureInput = document.getElementById("temperatureInput");
    let rainfallInput = document.getElementById("rainfallInput");
    let submitButton = document.getElementById("submitButton");

    if (predictionType === "rainfall") {
        temperatureInput.style.display = "block";
        rainfallInput.style.display = "none";
        submitButton.value = "Predict Rainfall";
    } else {
        temperatureInput.style.display = "none";
        rainfallInput.style.display = "block";
        submitButton.value = "Predict Temperature";
    }
}