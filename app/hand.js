import { createCard } from "./card.js";


export function removeCardFromHand(name) {
    const hand = document.getElementById("hand");
    let cardwrap;
    const cardnames = hand.querySelectorAll(".cardname");
    if (cardnames.length === 0){
        return;
    }
    for (const cardname of cardnames) {
        if (cardname.innerText === name) {
            cardwrap = cardname.parentElement;
            break;
        }
    }
    if (!cardwrap){
        console.error("All cards in Hand:", cardnames); // ####TODO to be deleted
        console.error(`Card ${name} not found in hand`);
    } else {
        hand.removeChild(cardwrap);
    }
}

export function displayHand(names) {
    // Create card elements
    const cardwraps = [];
    for (const name of names) {
        const cardwrap = createCard(name);
        cardwraps.push(cardwrap);
    }
    // TODO: add event listener to each cardwrap & add cardmenu
    // Replace hand
    const hand = document.getElementById("hand");
    hand.replaceChildren(...cardwraps);
}

export function displayTrade(coins) {
    const trades = [];
    for (const coin of coins){
        const trade = document.createElement("div");
        trade.className = "trade";
        trade.innerHTML = "";
        trade.innerHTML += `<span class="coin"></span>&nbsp;${coin[0]}`;
        trade.innerHTML += "&nbsp;".repeat(5);
        trade.innerHTML += `<span class="coin"></span>&nbsp;${coin[1]}`;
        trade.addEventListener("click", (e) => {
            e.stopPropagation();
            const event = new CustomEvent("trade", { detail: { trade: coin } });
            trade.dispatchEvent(event);
        });
        trades.push(trade);
    }
    // Add trade options to hand and mask all cardwraps in hand with display:none
    const cardwraps = document.querySelectorAll("#hand > .cardwrap");
    for (const cardwrap of cardwraps) {
        cardwrap.style.display = "none";
    }
    // Add trades to hand
    const hand = document.getElementById("hand");
    trades.forEach(trade => hand.appendChild(trade));
    /*const hand = document.getElementById("hand");
    hand.replaceChildren(...trades);*/
}

export function displayScore(task) {
    const keys = ["conflict", "wealth", "wonder", "civilian", "science", "commerce", "guild", "total"];
    // create HTML score table
    const scoretable = document.createElement("table");
    // insert header row "users"
    const headerRow = scoretable.insertRow();
    headerRow.classList.add("header-row");
    const headerCell = document.createElement("th");
    headerRow.appendChild(headerCell);
    task.users.forEach(user => {
        const cell = document.createElement("th");
        cell.textContent = user;
        headerRow.appendChild(cell);
    });
    // insert rows
    scoretable.classList.add("scoretable");
    for (const key of keys) {
        const row = scoretable.insertRow();
        const schemaCell = document.createElement("th");
        /*
        schemaCell.textContent = key.toUpperCase();
        row.appendChild(schemaCell);
        const values = task[key];
        values.forEach(value => {
            const cell = row.insertCell();
            cell.textContent = value;
        });
        */
        if (key === "total") {
            schemaCell.textContent = "TOTAL (COIN)";
            row.appendChild(schemaCell);

            const totalScores = task["total"];
            const coinScores = task["coin"];

            // Loop by index to combine the total and coin values
            for (let i = 0; i < task.users.length; i++) {
                const cell = row.insertCell();
                cell.textContent = `${totalScores[i]} (${coinScores[i]})`;
            }
        } else {
            schemaCell.textContent = key.toUpperCase();
            row.appendChild(schemaCell);
            
            const values = task[key];
            values.forEach(value => {
                const cell = row.insertCell();
                cell.textContent = value;
            });
        }
    }
    const hand = document.getElementById("hand");
    hand.appendChild(scoretable);
}