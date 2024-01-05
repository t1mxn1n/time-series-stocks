function functionToExecute(stock, period, interval, method) {
    console.log(stock, period, interval, method);

    let request = new XMLHttpRequest();

    let path = `http://127.0.0.1:8000/predict?stock=${stock}&period=${period}&interval=${interval}&method=${method}`;
    request.open('GET', path, true)
    request.onload = function () {
        if (request.status >= 200 && request.status < 400) {
            let response = JSON.parse(request.responseText);
            console.log(response);
            let imgElement = document.getElementById("img");
            let path = response["url"]["link"];
            imgElement.src = path;
            let imgDiv = document.getElementById("img_div");
            imgDiv.style.display = "block";

            let det = document.getElementById("details");

            det.innerHTML = `Затраченное время: ${response["time_processing"]}; Обучающая/Валидационная выборка: ${response["details"]["shape_training_set"]} | ${response["details"]["shape_validation_set"]}; RMS: ${response["details"]["rms"]}`;
            det.style.display = "block";

        } else {
            console.log('error button')
        }
    }
    request.send()

}

function functionToExecuteLSTM(stock, period, interval, future, epoch, batch) {
    console.log(stock, period, interval, future, epoch, batch);

    let request = new XMLHttpRequest();
    let path = `http://127.0.0.1:8000/lstm?stock=${stock}&period=${period}&interval=${interval}&need_future_predict=${future}&epoch=${epoch}&batch_size=${batch}`;
    request.open('GET', path, true)
    request.onload = function () {
        if (request.status >= 200 && request.status < 400) {
            let response = JSON.parse(request.responseText);
            console.log(response);
            let imgElement = document.getElementById("img");
            let path = response["url"]["link"];
            imgElement.src = path;
            let imgDiv = document.getElementById("img_div");
            imgDiv.style.display = "block";

            let det = document.getElementById("details");

            det.innerHTML = `Затраченное время: ${response["time_processing"]}; Обучающая/Валидационная выборка: ${response["details"]["shape_training_set"]} | ${response["details"]["shape_validation_set"]}; RMS: ${response["details"]["rms"]}`;
            det.style.display = "block";

        } else {
            console.log('error button')
        }
    }
    request.send()

}

