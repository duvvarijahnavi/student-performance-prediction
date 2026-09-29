const form = document.getElementById("predictionForm");
const resultContent = document.getElementById("resultContent");

form.addEventListener("submit", async function (event) {

    event.preventDefault();

    resultContent.innerHTML = `
        <p>⏳ Predicting student performance...</p>
    `;

    const studentData = {
        school: document.getElementById("school").value,
        sex: document.getElementById("sex").value,
        age: Number(document.getElementById("age").value),
        address: document.getElementById("address").value,
        famsize: document.getElementById("famsize").value,
        Pstatus: document.getElementById("Pstatus").value,

        Medu: Number(document.getElementById("Medu").value),
        Fedu: Number(document.getElementById("Fedu").value),

        Mjob: document.getElementById("Mjob").value,
        Fjob: document.getElementById("Fjob").value,
        reason: document.getElementById("reason").value,
        guardian: document.getElementById("guardian").value,

        traveltime: Number(document.getElementById("traveltime").value),
        studytime: Number(document.getElementById("studytime").value),
        failures: Number(document.getElementById("failures").value),

        schoolsup: document.getElementById("schoolsup").value,
        famsup: document.getElementById("famsup").value,
        paid: document.getElementById("paid").value,
        activities: document.getElementById("activities").value,
        nursery: document.getElementById("nursery").value,
        higher: document.getElementById("higher").value,
        internet: document.getElementById("internet").value,
        romantic: document.getElementById("romantic").value,

        famrel: Number(document.getElementById("famrel").value),
        freetime: Number(document.getElementById("freetime").value),
        goout: Number(document.getElementById("goout").value),
        Dalc: Number(document.getElementById("Dalc").value),
        Walc: Number(document.getElementById("Walc").value),
        health: Number(document.getElementById("health").value),

        absences: Number(document.getElementById("absences").value),

        G1: Number(document.getElementById("G1").value),
        G2: Number(document.getElementById("G2").value)
    };

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    student: studentData
                })
            }
        );

        const result = await response.json();

        if (!response.ok) {
            throw new Error(
                result.detail || "Prediction failed"
            );
        }

        resultContent.innerHTML = `
            <p>Predicted Final Grade</p>

            <div class="prediction-score">
                ${result.predicted_grade} / 20
            </div>

            <p class="prediction-message">
                ${getPerformanceMessage(result.predicted_grade)}
            </p>

            <p>
                Model: ${result.model_name}
            </p>

            <p>
                Expected MAE:
                ${result.expected_test_mae.toFixed(2)}
            </p>
        `;

    } catch (error) {

        console.error(error);

        resultContent.innerHTML = `
            <p style="color: red;">
                ❌ ${error.message}
            </p>
        `;
    }
});


function getPerformanceMessage(grade) {

    if (grade >= 16) {
        return "🌟 Excellent Performance";
    }

    if (grade >= 12) {
        return "👍 Good Performance";
    }

    if (grade >= 10) {
        return "✅ Likely to Pass";
    }

    return "⚠️ Needs Improvement";
}