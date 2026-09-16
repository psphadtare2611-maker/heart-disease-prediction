(function () {
  "use strict";

  var form = document.getElementById("predict-form");
  var submitBtn = document.getElementById("submit-btn");
  var resetBtn = document.getElementById("reset-btn");

  if (!form) return;

  form.addEventListener("submit", function (event) {
    if (!form.checkValidity()) {
      event.preventDefault();
      event.stopPropagation();
      form.classList.add("was-validated");
      return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "Analyzing...";
  });

  if (resetBtn) {
    resetBtn.addEventListener("click", function () {
      form.classList.remove("was-validated");
      submitBtn.disabled = false;
      submitBtn.textContent = "Predict Heart Disease Risk";
    });
  }
})();
