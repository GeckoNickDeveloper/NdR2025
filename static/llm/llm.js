/*
 * Initialize the script for the whole page.
 */
function init_script() {
    // Initialize the plot
    create_plot();

    // Set event listener for the chat input
    document.getElementById('chat-input').addEventListener('keyup', (e) => {
        // Process input text when pressing enter
        if (e.key === 'Enter') {
            process_text();
        }
    });
}

class MissingMaskError extends Error {
    constructor() {
        super("input text is missing [MASK]");
    }
}

/*
 * Query the server for an object with predicted tokens.
 */
async function fetch_tokens(text) {
    const url = "http://127.0.0.1:4200/api/llm"

    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: text,
    });

    if (!res.ok) {
        if (res.status == 418) {
            throw new MissingMaskError();
        } else {
            throw new Error(`${res.status} - ${res.statusText}`);
        }
    }

    // Get JSON response
    const response = await res.json();
    return response;
}

/*
 * Process the text inserted by the user in the chat input.
 *
 * This function is responsible of processing the text
 * inserted by the user into the chat box calling the
 * LLM api and displaying the tokens and the chart.
 *
 * Note: If the input text is empty the processing is
 * not performed.
 */
async function process_text() {
    let text = document.getElementById('chat-input').value;
    // text = "Lorem ipsum dolor sit amet, [MASK] adipiscing elit."
    console.log(`input: ${text}`);

    // skip processing empty text
    if (!text || text.length === 0) {
        return
    }

    try {
        // get tokens from server
        const response = await fetch_tokens(text);
        const tokens = response['tokens'];
        const pred_tokens = response['predictions'];

        // assign color to tokens
        tokens.forEach((e) => { e.color = get_token_color(e) });
        pred_tokens.forEach((e) => { e.color = get_token_color(e) });

        // sort predictions by confidence
        pred_tokens.sort((a, b) => {
            if (a.confidence < b.confidence) {
                return -1;
            } else if (a.confidence > b.confidence) {
                return 1;
            } else {
                return 0;
            }
        });
        pred_tokens.reverse();

        // display tokens and predictions
        display_tokens(tokens);
        update_plot(pred_tokens);

        // reset the input text
        document.getElementById('chat-input').value = "";
    } catch (error) {
        console.error(error);

        if (error instanceof MissingMaskError) {
            alert('Il testo deve contenere la maschera [MASK]');
        } else {
            alert(`Errore sconosciuto ${error}`);
        }
    }
}

/*
 * Return the color associated to the given token.
 */
let tokens_color_map = {}
function get_token_color(token) {
    if (token.text in tokens_color_map) {
        return tokens_color_map[token.text];
    }
    const c = get_random_color();
    tokens_color_map[token.text] = c;
    return c;
}

/*
 * Return a random good-looking color as an RGB hex string.
 */
function get_random_color() {
    // Code stolen from here: https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically
    const golden_ratio_conjugate = 0.618033988749895;
    let h = Math.random();
    h += golden_ratio_conjugate;
    h %= 1;
    return hsv_to_rgb(h, 0.60, 0.75);
}

/*
 * Convert a color expressed in HSV into a RGB hex string.
 */
function hsv_to_rgb(h, s, v) {
    let r, g, b;
    let i, f, p, q, t;

    i = Math.floor(h * 6);
    f = h * 6 - i;
    p = v * (1 - s);
    q = v * (1 - f * s);
    t = v * (1 - (1 - f) * s);

    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }
    r = Math.round(r * 255);
    g = Math.round(g * 255);
    b = Math.round(b * 255);

    // Format RGB hex string
    r = r.toString(16);
    g = g.toString(16);
    b = b.toString(16);
    if (r.length == 1)
        r = "0" + r;
    if (g.length == 1)
        g = "0" + g;
    if (b.length == 1)
        b = "0" + b;
    return "#" + r + g + b;
}

/* 
 * Show a given list of tokens into the page.
 *
 * Parameters:
 *   tokens: list of objects with: `id`, `text`, `color`.
 */
function display_tokens(tokens) {
    const tokens_element = document.getElementById('tokens-list');
    tokens_element.innerHTML = "";

    tokens.forEach((token) => {
        const t_container = document.createElement('div');
        t_container.className = 'token-container'
        t_container.style.background = token.color;

        const t_text = document.createElement('p');
        t_text.className = 'token-text';
        t_text.textContent = token.text;

        const t_id = document.createElement('p');
        t_id.className = 'token-id';
        t_id.textContent = token.id;

        t_container.appendChild(t_text);
        t_container.appendChild(t_id);
        tokens_element.appendChild(t_container);
    });
}

/*
* Create the plot without setting any data.
*/
let plot_chart;
function create_plot() {
    const plot_canvas = document.getElementById('plot-canvas')
    // Hide plot by default
    plot_canvas.style.visibility = 'hidden';

    const config = {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Confidenza',
                data: [],
                borderRadius: 5
            }]
        },
        options: {
            indexAxis: 'y',
            scales: {
                x: {
                    type: 'linear',
                    min: 0,
                    max: 100,
                    ticks: {
                        font: {
                            size: 20
                        }
                    }
                },
                y: {
                    ticks: {
                        font: {
                            size: 20
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        font: {
                            size: 25
                        }
                    },
                    display: true
                }
            },
            responsive: true,
            maintainAspectRatio: false
        }
    }
    plot_chart = new Chart(plot_canvas, config);
}

/*
 * Update the plot with the given predicted tokens.
 */
function update_plot(pred_tokens) {
    // Make plot visible
    document.getElementById('plot-canvas').style.visibility = 'visible';

    const labels = pred_tokens.map((e) => { return e.text });
    const data = pred_tokens.map((e) => { return e.confidence });
    const color = pred_tokens.map((e) => { return e.color });
    plot_chart.data.labels = labels;
    plot_chart.data.datasets[0].data = data;
    plot_chart.data.datasets[0].backgroundColor = color;
    plot_chart.update();
}
