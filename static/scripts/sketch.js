/* =================== Global */
/* ======= Elements */
var canvas = undefined;
var ctx = undefined;
var clearBtn = undefined;

/* ======= Variables */
// Flag for periodic inference request
var sendFlag = false;

// Last known position
var pos = { x: 0, y: 0 };

var offsets = { x: 0, y: 0 };


/* =================== Functions */
// Init
function init() {
    console.info('init')

    // Elements init
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    clearBtn = document.getElementById('clear');

    resize(); // Must

    // Elements event callbacks
    clearBtn.addEventListener('click', clearCanvas)

    // Variables
    const rect = canvas.getBoundingClientRect();
    offsets.x = rect.left
    offsets.y = rect.top

    // Drawing
    window.addEventListener('resize', resize);
    document.addEventListener('mousemove', draw);
    document.addEventListener('mousedown', setPosition);
    document.addEventListener('mouseenter', setPosition);

    // Set intervall
    setInterval(inferenceCallback, 600);
    setInterval(standbyCallback, 5_000); // Set send to false after 10s
}







// Request data
async function requestInference(img) {
    const url = "http://127.0.0.1:5000/api/sketch"

    try {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "image/jpeg" },
            body: img,
        });

        if (!res.ok) {
            throw new Error(`Response status: ${res.status}`);
        }

        // Get JSON response
        const response = await res.json();
        console.log(response);

        // Perform actions
            // Draw chart
            // Write text

    } catch (error) {
        console.error(error.message);
    }
}





// new position from mouse event
function setPosition(e) {
    // Removing offsets
    pos.x = e.clientX - offsets.x;
    pos.y = e.clientY - offsets.y;
}

// Resize canvas
function resize() {
    ctx.canvas.width = window.innerWidth;
    ctx.canvas.height = window.innerHeight;
}

// Draw on canvas
function draw(e) {
    // mouse left button must be pressed
    if (e.buttons !== 1) return;

    sendFlag = true

    ctx.beginPath(); // begin

    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#1f1f1f';

    ctx.moveTo(pos.x, pos.y); // from
    setPosition(e);
    ctx.lineTo(pos.x, pos.y); // to

    ctx.stroke(); // draw it!
}


// Clear canvas
function clearCanvas() {
    console.log('clear')
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}


// Timed callback
function inferenceCallback() {
    // console.info('[Inference] Timed callback')
    
    if (sendFlag) {
        console.info('Timed callback - Sending')
        
        canvas.toBlob((blob) => requestInference(blob), "image/png")
        // img = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        

        // // Send
        // requestInference(img)
    }
}

function standbyCallback() {
    // console.info('[Stand-By] Timed callback')
    
    sendFlag = false
}