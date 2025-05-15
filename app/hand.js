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
