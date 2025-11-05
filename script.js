// CyberCrawl Spider Robot - Frontend JavaScript

let currentMode = 'STOPPED';
let statusUpdateInterval = null;
let isInitialized = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('🕷️ CyberCrawl Interface Loaded');
    initializeUI();
    startStatusUpdates();
});

function initializeUI() {
    console.log('⚙️ Initializing UI...');
    updateUIState();
    isInitialized = true;
}

// ===== Status Updates =====
function startStatusUpdates() {
    statusUpdateInterval = setInterval(updateStatus, 1000);
    updateStatus(); // Initial update
}

async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        currentMode = data.mode;
        updateUIState();
        updateDistance(data.distance);
        updateDetections(data.detections);
        updatePerformance(data);
        
    } catch (error) {
        console.error('❌ Status update error:', error);
    }
}

function updateUIState() {
    const statusBadge = document.getElementById('statusBadge');
    const statusText = document.querySelector('.status-text');
    const modeText = document.getElementById('modeText');
    const manualControls = document.getElementById('manualControls');
    
    const btnAutoMode = document.getElementById('btnAutoMode');
    const btnStop = document.getElementById('btnStop');
    const btnManualMode = document.getElementById('btnManualMode');
    
    // Clear previous classes
    statusBadge.className = 'status-badge';
    
    if (currentMode === 'AUTO') {
        statusBadge.classList.add('auto');
        statusText.textContent = '🤖 AUTO MODE';
        modeText.textContent = 'AUTO';
        
        btnAutoMode.disabled = true;
        btnStop.disabled = false;
        btnManualMode.disabled = true;
        manualControls.style.display = 'none';
        
    } else if (currentMode === 'MANUAL') {
        statusBadge.classList.add('manual');
        statusText.textContent = '⚙️ MANUAL MODE';
        modeText.textContent = 'MANUAL';
        
        btnAutoMode.disabled = true;
        btnStop.disabled = false;
        btnManualMode.disabled = true;
        manualControls.style.display = 'block';
        
    } else {
        statusText.textContent = '⏸️ STOPPED';
        modeText.textContent = 'STOPPED';
        
        btnAutoMode.disabled = false;
        btnStop.disabled = true;
        btnManualMode.disabled = false;
        manualControls.style.display = 'none';
    }
}

function updateDistance(distance) {
    const distanceValue = document.querySelector('.overlay-info .distance .value');
    const distanceText = document.getElementById('distanceText');
    
    if (distance > 0) {
        const value = `${distance} cm`;
        distanceValue.textContent = value;
        distanceText.textContent = value;
        
        // Color code
        if (distance < 20) {
            distanceValue.style.color = '#ff006e';
        } else if (distance < 50) {
            distanceValue.style.color = '#ffa500';
        } else {
            distanceValue.style.color = '#06ffa5';
        }
    } else {
        distanceValue.textContent = '-- cm';
        distanceText.textContent = '-- cm';
    }
}

function updateDetections(detections) {
    const detectionCount = document.querySelector('.detection-badge .badge-value');
    const objectsText = document.getElementById('objectsText');
    const detectionsList = document.getElementById('detectionsList');
    
    const count = detections ? detections.length : 0;
    detectionCount.textContent = count;
    objectsText.textContent = count;
    
    // Update list
    detectionsList.innerHTML = '';
    if (detections && detections.length > 0) {
        detections.forEach(det => {
            const item = document.createElement('div');
            item.className = 'detection-item';
            const conf = Math.round(det.confidence * 100);
            item.innerHTML = `
                <span>${det.class}</span>
                <span class="detection-conf">${conf}%</span>
            `;
            detectionsList.appendChild(item);
        });
    } else {
        detectionsList.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">No objects detected</p>';
    }
}

function updatePerformance(data) {
    const fpsValue = document.querySelector('.overlay-info .fps .value');
    const fpsText = document.getElementById('fpsText');
    
    if (data.fps > 0) {
        fpsValue.textContent = `${data.fps} FPS`;
        fpsText.textContent = `${data.fps} fps`;
    } else {
        fpsValue.textContent = '-- FPS';
        fpsText.textContent = '-- fps';
    }
}

// ===== Control Functions =====
async function startAutoMode() {
    if (currentMode !== 'STOPPED') {
        showToast('⚠️ Please stop the robot first', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/start_auto', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showToast('🤖 Auto mode started!', 'success');
            currentMode = 'AUTO';
            updateUIState();
        } else {
            showToast('❌ ' + data.message, 'error');
        }
    } catch (error) {
        showToast('❌ Connection error', 'error');
        console.error('Error:', error);
    }
}

async function stopRobot() {
    try {
        const response = await fetch('/api/stop', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showToast('🛑 Robot stopped', 'success');
            currentMode = 'STOPPED';
            updateUIState();
        } else {
            showToast('❌ ' + data.message, 'error');
        }
    } catch (error) {
        showToast('❌ Connection error', 'error');
        console.error('Error:', error);
    }
}

async function enableManualMode() {
    if (currentMode !== 'STOPPED') {
        showToast('⚠️ Please stop the robot first', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/manual_mode', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showToast('⚙️ Manual mode activated!', 'success');
            currentMode = 'MANUAL';
            updateUIState();
        } else {
            showToast('❌ ' + data.message, 'error');
        }
    } catch (error) {
        showToast('❌ Connection error', 'error');
        console.error('Error:', error);
    }
}

async function manualControl(action) {
    if (currentMode !== 'MANUAL') {
        showToast('⚠️ Not in manual mode', 'error');
        return;
    }
    
    const actionText = {
        'forward': '⬆️ Forward',
        'backward': '⬇️ Backward',
        'left': '⬅️ Left',
        'right': '➡️ Right',
        'wave': '👋 Wave',
        'shake': '🤝 Shake',
        'dance': '💃 Dance',
        'stand': '🧍 Stand',
        'sit': '💺 Sit'
    };
    
    try {
        const response = await fetch(`/api/manual_control/${action}`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showToast(actionText[action], 'info');
        } else {
            showToast('❌ ' + data.message, 'error');
        }
    } catch (error) {
        showToast('❌ Connection error', 'error');
    }
}

// ===== Toast Notifications =====
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = 'toast show';
    
    if (type === 'error') {
        toast.classList.add('error');
    } else if (type === 'success') {
        toast.classList.add('success');
    }
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// ===== Keyboard Controls =====
document.addEventListener('keydown', function(event) {
    if (currentMode !== 'MANUAL') return;
    
    const key = event.key.toLowerCase();
    
    if (key === 'arrowup' || key === 'w') {
        event.preventDefault();
        manualControl('forward');
    } else if (key === 'arrowdown' || key === 's') {
        event.preventDefault();
        manualControl('backward');
    } else if (key === 'arrowleft' || key === 'a') {
        event.preventDefault();
        manualControl('left');
    } else if (key === 'arrowright' || key === 'd') {
        event.preventDefault();
        manualControl('right');
    } else if (key === ' ') {
        event.preventDefault();
        stopRobot();
    }
});

// ===== Cleanup =====
window.addEventListener('beforeunload', function() {
    if (statusUpdateInterval) {
        clearInterval(statusUpdateInterval);
    }
});

console.log('✅ CyberCrawl Controls Ready!');