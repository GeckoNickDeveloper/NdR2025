var canvas = undefined;
var ctx = undefined;
var clearBtn = undefined;

// Flag for periodic inference request
var sendFlag = false;

// Last known position
var pos = { x: 0, y: 0 };
var offsets = { x: 0, y: 0 };

/*
 * Initialize the script for the whole page.
 */
function init_script() {
    console.info('init');

    // Elements init
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    clearBtn = document.getElementById('clear');

    resize();

    // callbacks
    clearBtn.addEventListener('click', clearCanvas);
    window.addEventListener('resize', resize);
    document.addEventListener('mousemove', draw);
    document.addEventListener('mousedown', setPosition);
    document.addEventListener('mouseenter', setPosition);

    // Send drawings to the server every second
    setInterval(() => {
        if (sendFlag) {
            console.info('Timed callback - Sending');
            canvas.toBlob((blob) => process_drawing(blob), "image/jpeg");
        }
    }, 1_000);
    // Stop sending drawings to the server after two seconds of inactivity
    setInterval(() => {
        sendFlag = false;
    }, 2_000);
}

/*
 * Set new mouse position.
 */
function setPosition(e) {
    // Removing offsets
    pos.x = e.clientX - offsets.x;
    pos.y = e.clientY - offsets.y;
}

/*
 * Clear the canvas.
 */
function clearCanvas() {
    console.log('clear canvas')
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    document.getElementById('best-class-box').style.visibility = 'hidden';
}

/*
 * Resize the canvas.
 */
function resize() {
    const parent = document.getElementById('sketch-area');
    ctx.canvas.width = parent.clientWidth;
    ctx.canvas.height = parent.clientHeight;

    const rect = canvas.getBoundingClientRect();
    offsets.x = rect.left;
    offsets.y = rect.top;
    clearCanvas();
}

/*
 * Draw on the canvas.
 */
function draw(e) {
    // mouse left button must be pressed
    if (e.buttons !== 1) {
        return;
    }

    sendFlag = true;

    // begin
    ctx.beginPath();
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000' //'#1f1f1f';
    ctx.moveTo(pos.x, pos.y); // from
    setPosition(e);
    ctx.lineTo(pos.x, pos.y); // to
    ctx.stroke(); // draw it!
}

/*
 * Send drawing to the server for processing
 */
async function requestInference(img) {
    const url = "http://127.0.0.1:4200/api/sketch";

    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: img,
    });

    if (!res.ok) {
        throw new Error(`${res.status} - ${res.statusText}`);
    }

    // Get JSON response
    const response = await res.json();
    return response;
}

/*
 * Process the drawing.
 */
async function process_drawing(img) {
    try {
        const response = await requestInference(img);
        const top = response[0]['top'];

        const best_box = document.getElementById('best-class-box')
        best_box.style.visibility = 'visible';
        best_box.innerHTML = "";
        top.forEach((e, idx) => {
            const label = e['label'].toUpperCase();
            const conf = e['conf'].toFixed(2);
            const text_element = document.createElement('p');
            text_element.className = "result";
            text_element.textContent = `${label} - ${conf}%`;
            text_element.style.fontSize = `${24 - 3 * idx}px`;
            best_box.appendChild(text_element);
        });
    } catch (error) {
        console.error(error.message);
    }
}