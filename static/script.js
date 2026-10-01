// Client-side checks and a loading indicator for the upload form.
document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("#analyze-form");
    if (!form) return;

    form.addEventListener("submit", (event) => {
        const audio = document.querySelector("#audio");
        const text = document.querySelector("#transcript").value.trim();
        if (audio.files.length === 0 && text === "") {
            event.preventDefault();
            alert("Choose an audio file or paste a transcript first.");
            return;
        }
        // Long transcriptions look like a frozen page, so show progress
        document.querySelector("#submit-btn").disabled = true;
        document.querySelector("#working").classList.remove("d-none");
    });
});
