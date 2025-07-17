const recordBtn = document.getElementById("record-btn");
let mediaRecorder, audioChunks = [];

recordBtn.addEventListener("click", () => {
    // Access UI helper functions from the global scope (defined in index.html)
    const showSpinner = window.showSpinner;
    const showResult = window.showResult;

    if (!mediaRecorder || mediaRecorder.state === "inactive") {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                mediaRecorder.start();
                audioChunks = [];
                recordBtn.textContent = "⏹️ Stop Recording";
                recordBtn.classList.replace("btn-danger", "btn-warning");

                mediaRecorder.ondataavailable = e => {
                    audioChunks.push(e.data);
                };

                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' }); // Create a WAV blob
                    const formData = new FormData();
                    formData.append("file", audioBlob, "recording.wav"); // ✅ Send as a .wav file

                    showSpinner(); // Show spinner and status

                    fetch("/predict", { method: "POST", body: formData })
                        .then(res => res.json())
                        .then(data => {
                            showResult(data); // ✅ Use the helper to display result
                        })
                        .catch(err => {
                             showResult({ error: 'Network or server error.' });
                        });
                };
            })
            .catch(err => {
                showResult({ error: 'Could not access microphone.' });
            });
    } else if (mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recordBtn.textContent = "🎙️ Start Recording";
        recordBtn.classList.replace("btn-warning", "btn-danger");
    }
});