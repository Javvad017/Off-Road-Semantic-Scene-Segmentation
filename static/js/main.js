const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const youtubeUrl = document.getElementById('youtube-url');
const processBtn = document.getElementById('btn-process');
const statusPanel = document.getElementById('status-panel');
const resultsPanel = document.getElementById('results-panel');
const progressBar = document.getElementById('progress-bar');
const statusText = document.getElementById('status-text');
const statusPercent = document.getElementById('status-percent');
const inputVideo = document.getElementById('input-video');
const outputVideo = document.getElementById('output-video');
const downloadLink = document.getElementById('download-link');

// Drag & Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

processBtn.addEventListener('click', () => {
    const url = youtubeUrl.value.trim();
    if (url) {
        handleYoutube(url);
    }
});

// Handlers
function handleFile(file) {
    statusPanel.style.display = 'block';
    resultsPanel.style.display = 'none';
    
    // Create FormData
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.task_id) {
            pollStatus(data.task_id);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(err => {
        console.error(err);
        alert('Upload failed.');
    });
}

function handleYoutube(url) {
    statusPanel.style.display = 'block';
    resultsPanel.style.display = 'none';

    const formData = new FormData();
    formData.append('youtube_url', url);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.task_id) {
            pollStatus(data.task_id);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(err => {
        console.error(err);
        alert('Validation failed.');
    });
}

function pollStatus(taskId) {
    const interval = setInterval(() => {
        fetch(`/status/${taskId}`)
        .then(response => response.json())
        .then(data => {
            // Update Progress
            const progress = data.progress || 0;
            progressBar.style.width = `${progress}%`;
            statusPercent.innerText = `${progress}%`;
            statusText.innerText = data.status;

            if (data.state === 'COMPLETED' || data.state === 'SUCCESS') {
                clearInterval(interval);
                showResults(data);
            } else if (data.state === 'FAILED') {
                clearInterval(interval);
                alert('Processing Failed: ' + data.error);
                statusPanel.style.display = 'none';
            }
        });
    }, 1000);
}

function showResults(data) {
    statusPanel.style.display = 'none';
    resultsPanel.style.display = 'flex';
    
    // Update Videos
    // Using cache-busting timestamp
    const ts = new Date().getTime();
    inputVideo.src = `${data.input_url}?t=${ts}`;
    outputVideo.src = `${data.output_url}?t=${ts}`;
    
    // Loop
    inputVideo.play();
    outputVideo.play();
    
    // Update Stats
    document.getElementById('fps-val').innerText = data.fps || '--';
    document.getElementById('safe-val').innerText = '100 Check';

    // Download
    downloadLink.href = data.download_url;
}
