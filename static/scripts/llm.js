

// Request data
async function requestInference() {
    const url = "http://127.0.0.1:5000/api/llm"

    // Get text from input field
    const text = document.getElementById('chat-input').value;
    console.warn(text);

    try {
        const res = await fetch(url, {
            method: "POST",
            body: text,
            headers: {
                "Content-Type": "plain/text",
            },
        });

        if (!res.ok) {
            throw new Error(`Response status: ${response.status}`);
        }

        // Get JSON response
        const response = await res.json();
        console.log(result);

        // Perform actions
        show_tokens('tokens', response['tokens'])
        plot_hist('plot-canvas', response['out-tokens'], response['conf'])

    } catch {
        console.error(error.message);
    }
}






















function req_inference(id) {
    let text = document.getElementById(id).value
    console.warn(text)

//     window.setTimeout(() => {
//         let res = {
//             tokens: ['asdf', 'asdf', 'asdf', 'asdf', 'asdf', 'asdf', 'asdf', 'asdf', 'asdf'],
//             labels: ['a','b','c','d','e','f'],
//             conf: [12,23,43,65,2],
//         }
//
//         show_tokens('tokens', res.tokens)
//         plot_hist('plot-canvas', res.labels, res.conf)
//     }, 1000)
//
    let xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            console.log(xmlHttp.responseText)
            var res = JSON.parse(xmlHttp.responseText)

            
    }
    xmlHttp.open("GET", "http://172.0.0.1:5000/predict", true); // true for asynchronous
    xmlHttp.send(text);
}



function randColor() {
    let letters = '6789ABCDEF'
    let color = '#'

    for(let i = 0; i < 6; ++i) {
        color += letters[Math.floor(Math.random() * 10)]
    }

    return color
}


function show_tokens(id, tokens) {
    const ctx = document.getElementById(id)

    tokens.forEach(t => {
        let item = `<span style="background-color:${randColor()}"> ${t} </span> `
        
        ctx.innerHTML += item
    });
}

function plot_hist(id, labels, data) {
    const ctx = document.getElementById(id)
    const cfg = {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidenza',
                data: data
            }]
        },
        options: {
            scales: {
                y: {
                    type: 'linear',
                    min: 0,
                    max: 100
                }
            },
            responsive: true,
            maintainAspectRatio: false,
            
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    }

    new Chart(ctx, cfg)
}
