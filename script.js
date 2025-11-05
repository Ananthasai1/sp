// CyberCrawl Spider Robot - Frontend JavaScript

let currentMode = 'STOPPED';
let statusUpdateInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('🕷️ CyberCrawl Interface Loaded');
    startStatusUpdates();
    updateUIState();
    
    // Refresh video feed after 5 seconds to ensure it starts displaying
    setTimeout(function() {
        const videoFeed = document.getElementById('videoFeed');
        if (videoFeed) {
            const currentSrc = videoFeed.src;
            videoFeed.src = currentSrc + '?t=' + Date.now();
        }
    }, 5000);
});

// ===== Status Updates =====
function startStatusUpdates() {
    // Update status every 500ms
    statusUpdateInterval = setInterval(updateStatus, 500);
}

async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        // Update mode
        currentMode = data.mode;
        updateUIState();
        
        // Update distance
        updateDistance(data.distance);
        
        // Update detections
        updateDetections(data.detections);
        
        // Update FPS
        const fpsDisplay = document.getElementById('fpsDisplay');
        if (fpsDisplay && data.fps) {
            fpsDisplay.textContent = data.fps + ' fps';
        }
        
    } catch (error) {
        console.error('Status update error:', error);
    }
}

function updateUIState() {
    // Update status badge
    const statusBadge = document.getElementById('statusBadge');
    const statusText = document.querySelector('.status-text');
    const modeText = document.getElementById('modeText');
    
    statusBadge.className = 'status-badge';
    
    if (currentMode === 'AUTO') {
        statusBadge.classList.add('auto');
        statusText.textContent = '🤖 AUTO MODE';
        modeText.textContent = 'AUTO';
    } else if (currentMode === 'MANUAL') {
        statusBadge.classList.add('manual');
        statusText.textContent = '⚙️ MANUAL MODE';
        modeText.textContent = 'MANUAL';
    } else {
        statusBadge.classList.add('stopped');
        statusText.textContent = '⏸️ STOPPED';
        modeText.textContent = 'STOPPED';
    }
    
    // Update button states
    const btnAutoMode = document.getElementById('btnAutoMode');
    const btnStop = document.getElementById('btnStop');
    const btnManualMode = document.getElementById('btnManualMode');
    const manualControls = document.getElementById('manualControls');
    
    if (currentMode === 'STOPPED') {
        btnAutoMode.disabled = false;
        btnStop.disabled = true;
        btnManualMode.disabled = false;
        manualControls.style.display = 'none';
    } else if (currentMode === 'AUTO') {
        btnAutoMode.disabled = true;
        btnStop.disabled = false;
        btnManualMode.disabled = true;
        manualControls.style.display = 'none';
    } else if (currentMode === 'MANUAL') {
        btnAutoMode.disabled = true;
        btnStop.disabled = false;
        btnManualMode.disabled = true;
        manualControls.style.display = 'block';
    }
}

function updateDistance(distance) {
    const distanceValue = document.querySelector('.distance-value');
    const distanceText = document.getElementById('distanceText');
    
    if (distance > 0) {
        distanceValue.textContent = `${distance} cm`;
        distanceText.textContent = `${distance} cm`;
        
        // Color code based on distance
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
        distanceValue.style.color = '#a0a0b0';
    }
}

function updateDetections(detections) {
    const detectionCount = document.getElementById('detectionCount');
    const detectionsList = document.getElementById('detectionsList');
    const detectionsText = document.getElementById('detectionsText');
    
    // Update count
    const count = detections.length;
    detectionCount.innerHTML = `<span class="badge">${count} ${count === 1 ? 'object' : 'objects'}</span>`;
    detectionsText.textContent = count;
    
    // Update list
    detectionsList.innerHTML = '';
    if (count === 0) {
        detectionsList.innerHTML = '<p style="color: #a0a0b0; font-size: 0.9rem;">No objects detected</p>';
    } else {
        detections.forEach(detection => {
            const item = document.createElement('div');
            item.className = 'detection-item';
            item.innerHTML = `
                <span>${detection.class}</span>
                <span class="detection-conf">${Math.round(detection.confidence * 100)}%</span>
            `;
            detectionsList.appendChild(item);
        });
    }
}

// ===== Control Functions =====
async function startAutoMode() {
    if (currentMode !== 'STOPPED') {
        showToast('Please stop the robot first', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/start_auto', {
            method: 'POST'
        });
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
        const response = await fetch('/api/stop', {
            method: 'POST'
        });
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
        showToast('Please stop the robot first', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/manual_mode', {
            method: 'POST'
        });
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
        showToast('Not in manual mode', 'error');
        return;
    }
    
    // Show action feedback
    const actionText = {
        'forward': '⬆️ Moving forward...',
        'backward': '⬇️ Moving backward...',
        'left': '⬅️ Turning left...',
        'right': '➡️ Turning right...',
        'wave': '👋 Waving...',
        'shake': '🤝 Shaking hand...',
        'dance': '💃 Dancing...',
        'stand': '🧍 Standing...',
        'sit': '💺 Sitting...'
    };
    
    showToast(actionText[action] || 'Executing...', 'info');
    
    try {
        const response = await fetch(`/api/manual_control/${action}`, {
            method: 'POST'
        });
        const data = await response.json();
        
        if (!data.success) {
            showToast('❌ ' + data.message, 'error');
        }
    } catch (error) {
        showToast('❌ Connection error', 'error');
        console.error('Error:', error);
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

// ===== Keyboard Controls (for manual mode) =====
document.addEventListener('keydown', function(event) {
    if (currentMode !== 'MANUAL') return;
    
    switch(event.key) {
        case 'ArrowUp':
        case 'w':
        case 'W':
            event.preventDefault();
            manualControl('forward');
            break;
        case 'ArrowDown':
        case 's':
        case 'S':
            event.preventDefault();
            manualControl('backward');
            break;
        case 'ArrowLeft':
        case 'a':
        case 'A':
            event.preventDefault();
            manualControl('left');
            break;
        case 'ArrowRight':
        case 'd':
        case 'D':
            event.preventDefault();
            manualControl('right');
            break;
        case ' ':
            event.preventDefault();
            stopRobot();
            break;
    }
});

// ===== Cleanup on page unload =====
window.addEventListener('beforeunload', function() {
    if (statusUpdateInterval) {
        clearInterval(statusUpdateInterval);
    }
});

console.log('🎮 Controls ready! Use arrow keys or WASD in manual mode.');