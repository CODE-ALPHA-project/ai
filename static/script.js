document.getElementById("question-form").addEventListener("submit", function (event) {
    event.preventDefault();
    const question = document.getElementById("question").value;
    document.getElementById("loading").style.display = "block";
    document.getElementById("answer").innerHTML = "";

    fetch("/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: question }),
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById("loading").style.display = "none";
            document.getElementById("answer").innerHTML = data.answer;
        })
        .catch(error => {
            document.getElementById("loading").style.display = "none";
            document.getElementById("answer").innerHTML = "오류가 발생했습니다. 다시 시도하세요.";
        });
});