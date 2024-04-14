document.addEventListener("DOMContentLoaded", function () {
  var form = document.getElementById("videoUploadForm");
  var submitButton = form.querySelector('button[type="submit"]'); // Get the submit button

  form.onsubmit = function (event) {
    event.preventDefault(); // Stop the form from causing a page refresh.
    submitButton.disabled = true; // Disable the button when the form is submitted

    var formData = new FormData(form);

    fetch("http://127.0.0.1:8000/api/upload/", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        // Handle success data here - maybe display a success message
      })
      .catch((error) => {
        console.error("Error:", error);
        // Handle errors here, such as displaying an error message
      })
      .finally(() => {
        submitButton.disabled = false; // Re-enable the button after the fetch is complete
      });
  };
});
