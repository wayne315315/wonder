export const CARDS = await (await fetch("../card.json")).json();


export function createCard(name) {
    // create card wrap element
    const cardWrap = document.createElement("div");
    cardWrap.className = "cardwrap";
    // create card element
    const id = CARDS[name]["id"];
    const bg_pos_y = Math.floor(id / 10) * -100;
    const bg_pos_x = (id % 10) * -100;
    const card = document.createElement("div");
    card.className = "card";
    card.style.backgroundPosition = `${bg_pos_x}% ${bg_pos_y}%`;
    // create cardname element
    const cardName = document.createElement("Span");
    cardName.className = "cardname";
    cardName.innerText = name;
    for (const action of ["build", "wonder", "discard"]) {
        // create cardmenu element
        const cardMenu = document.createElement("div");
        cardMenu.className = "cardmenu";
        cardMenu.classList.add(`cardmenu--${action}`);
        const cardArrow = document.createElement("div");
        cardArrow.className = "cardarrow";
        const cardOption = document.createElement("div");
        cardOption.className = "cardoption";
        cardOption.classList.add(`cardoption--${action}`);
        // add event listener to cardmenu
        cardMenu.addEventListener("click", (e) => {
            e.stopPropagation();
            const event = new CustomEvent("move", { detail: { pick: name, action: action.toUpperCase() } });
            cardMenu.dispatchEvent(event);
            console.log(`cardmenu clicked: ${event.detail.pick} ${event.detail.action}`);
        });
        // add card arrow and card option to cardmenu
        cardMenu.appendChild(cardArrow);
        cardMenu.appendChild(cardOption);
        cardWrap.appendChild(cardMenu);
    }
    // attach card, cardname to cardwrap
    cardWrap.appendChild(card);
    cardWrap.appendChild(cardName);
    //cardWrap.appendChild(cardMenu);
    return cardWrap;
}

export function createCardback(age) {
    const cardback = document.createElement("div");
    cardback.className = "cardback";
    cardback.classList.add("cardback--age_" + age);
    return cardback;
}
