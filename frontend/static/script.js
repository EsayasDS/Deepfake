document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const scanningState = document.getElementById('scanning-state');
    const resultsSection = document.getElementById('results-section');
    const previewImage = document.getElementById('preview-image');
    const previewVideo = document.getElementById('preview-video');
    const verdictBadge = document.getElementById('verdict-badge');
    const confidenceValue = document.getElementById('confidence-value');
    const progressFill = document.getElementById('progress-fill');
    const statThreshold = document.getElementById('stat-threshold');
    const statScore = document.getElementById('stat-score');
    const statFrames = document.getElementById('stat-frames');
    const framesBox = document.getElementById('frames-box');
    const resetBtn = document.getElementById('reset-btn');
    const verdictCard = document.getElementById('verdict-card');

    // Mobile menu
    const mobileBtn = document.getElementById('mobile-menu-btn');
    const navLinks = document.getElementById('nav-links');
    if (mobileBtn) {
        mobileBtn.addEventListener('click', () => navLinks.classList.toggle('open'));
    }
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => navLinks.classList.remove('open'));
    });

    // Drag & Drop
    ['dragenter','dragover','dragleave','drop'].forEach(e => {
        dropZone.addEventListener(e, ev => { ev.preventDefault(); ev.stopPropagation(); }, false);
    });
    ['dragenter','dragover'].forEach(e => {
        dropZone.addEventListener(e, () => dropZone.classList.add('dragover'));
    });
    ['dragleave','drop'].forEach(e => {
        dropZone.addEventListener(e, () => dropZone.classList.remove('dragover'));
    });
    dropZone.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
    fileInput.addEventListener('change', function() { handleFiles(this.files); });
    resetBtn.addEventListener('click', resetUI);

    function resetUI() {
        resultsSection.classList.add('hidden');
        scanningState.classList.add('hidden');
        dropZone.style.display = '';
        fileInput.value = '';
        previewImage.src = '';
        previewImage.classList.add('hidden');
        previewVideo.src = '';
        previewVideo.classList.add('hidden');
        framesBox.style.display = 'none';
        verdictCard.style.borderColor = '';
    }

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        const fileURL = URL.createObjectURL(file);
        if (file.type.startsWith('image/')) {
            previewImage.src = fileURL;
            previewImage.classList.remove('hidden');
            previewVideo.classList.add('hidden');
            uploadFile(file, '/api/analyze/image');
        } else if (file.type.startsWith('video/')) {
            previewVideo.src = fileURL;
            previewVideo.classList.remove('hidden');
            previewImage.classList.add('hidden');
            uploadFile(file, '/api/analyze/video');
        } else {
            alert('Please upload an image or video file.');
            return;
        }
        dropZone.style.display = 'none';
        scanningState.classList.remove('hidden');
        resultsSection.classList.add('hidden');
    }

    // --- CONFIGURATION ---
    // For local development, leave empty ('').
    // For Vercel production, replace with your Render URL (e.g., 'https://your-deepfake-api.onrender.com')
    const BACKEND_URL = ''; 
    // ---------------------

    function uploadFile(file, endpoint) {
        const formData = new FormData();
        formData.append('file', file);
        
        // Use the configured backend URL
        const fullUrl = BACKEND_URL + endpoint;

        fetch(fullUrl, { method: 'POST', body: formData })
            .then(res => res.json())
            .then(data => {
                scanningState.classList.add('hidden');
                if (data.error) { alert('Analysis error: ' + data.error); resetUI(); }
                else { showResults(data.result, data.type); }
            })
            .catch(err => {
                scanningState.classList.add('hidden');
                alert('Server error. Please ensure the backend is running.');
                resetUI();
            });
    }

    function showResults(result, type) {
        verdictBadge.textContent = result.verdict;
        verdictBadge.className = 'verdict-badge ' + result.verdict.toLowerCase();
        confidenceValue.textContent = result.confidence.toFixed(1) + '%';
        statThreshold.textContent = result.threshold.toFixed(4);
        statScore.textContent = result.score.toFixed(4);
        if (type === 'video' && result.frames_analyzed) {
            framesBox.style.display = '';
            statFrames.textContent = result.frames_analyzed;
        } else { framesBox.style.display = 'none'; }
        const isReal = result.verdict === 'REAL';
        progressFill.style.width = '0%';
        progressFill.style.background = isReal ? 'linear-gradient(90deg,#00ff88,#00ccff)' : 'linear-gradient(90deg,#ff3366,#ff6644)';
        progressFill.style.boxShadow = isReal ? '0 0 12px rgba(0,255,136,0.4)' : '0 0 12px rgba(255,51,102,0.4)';
        verdictCard.style.borderColor = isReal ? 'rgba(0,255,136,0.3)' : 'rgba(255,51,102,0.3)';
        resultsSection.classList.remove('hidden');
        setTimeout(() => { progressFill.style.width = Math.min(result.confidence, 100) + '%'; }, 100);
    }

    // Navbar scroll
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        navbar.style.background = window.scrollY > 50 ? 'rgba(6,6,10,0.97)' : 'rgba(6,6,10,0.9)';
    });

    // Smooth scroll
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) target.scrollIntoView({ behavior: 'smooth' });
        });
    });
});
