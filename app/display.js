import { createBoard, CIVS, FACES } from "./board.js";
import { displayHand, displayTrade, removeCardFromHand } from "./hand.js";
import { createCard, createCardback, CARDS } from "./card.js";


// task.type : "SETTING", "FACE", "AGE", "MOVE", "UPDATE", "TRADE", "CLEAR", "BATTLE", "SCORE"
export async function display(task) {
    const banner = document.getElementById("banner");
    const hand = document.getElementById("hand");
    switch (task.type){
        case "SETTING":
            const arena = document.getElementById("arena");
            while (arena.firstChild) {
                arena.removeChild(arena.firstChild);
            }
            // clean up banner
            banner.innerText = "";   
            // create board
            const civs = task.civs;
            const faces = task.faces;
            const n = civs.length;
            for (let i = 0; i < n; i++){
                let civ = civs[i];
                let face =  (faces) ? faces[i] : null;
                if (face) {
                    createBoard(i, n, civ, face);
                } else {                
                    createBoard(i, n, civ, "Night");
                    let boardback = document.getElementById(`player_board_group_${i}`).firstChild;
                    boardback.classList.add("boardback");
                    createBoard(i, n, civ, "Day");
                    // board group
                    let boardGroup = document.getElementById(`player_board_group_${i}`);
                    boardGroup.classList.add("board_group--face");
                }
            }
            break;
        case "FACE":
            banner.innerText = "Choose the side of your wonder board";
            for (const face of FACES) {
                const facebutton = document.createElement("div");
                facebutton.classList.add("facebutton");
                facebutton.classList.add(`facebutton--${face}`);
                facebutton.addEventListener("click", (e) => {
                    e.stopPropagation();
                    const event = new CustomEvent("face", { detail: { face: face } });
                    facebutton.dispatchEvent(event);
                });
                hand.appendChild(facebutton);
            };
            console.log("facebutton created");
            break;
        case "AGE":
            let age = task.age;
            banner.innerText = `Age ${"I".repeat(age)} started`;
            break;
        case "MOVE":
            banner.innerText = `Choose a card to play`;
            displayHand(task.hand);
            break;
        case "TRADE":
            banner.innerText = "Choose a trade option";
            displayTrade(task.coins);
            break;
        case "UPDATE":
            banner.innerText = "Update...";
            await displayUpdate(task.coins, task.moves, task.age, task.civs, task.faces);
            break;
        case "CLEAR":
            banner.innerText = "Discard all cards";
            while (hand.firstChild) {
                hand.removeChild(hand.firstChild);
            }
            break;
        case "BATTLE":
            // TODO
            banner.innerText = "BATTLE";
            break;
        case "SCORE":
            // TODO
            break;
    }
}

//parentElement.querySelectorAll('.removable').forEach(element => element.remove());
async function displayUpdate(coins, moves, age, civs, faces) {
    // Remove selected card from hand
    if (moves[0]){
        // remove all trades from hand and show all cards
        document.querySelectorAll(".trade").forEach(element => element.remove());
        document.querySelectorAll("#hand > .cardwrap").forEach(element => {
            element.style.display = "block";
        })
        // remove card from hand
        const playedcard = moves[0].pick;
        removeCardFromHand(playedcard);
    }
    // promises collection
    const promises = [];
    // display moves
    for (let i = 0; i < moves.length; i++) {
        promises.push(displayMove(i, moves[i], age, civs, faces));
    }
    // display coins
    for (let i = 0; i < coins.length; i++) {
        const coin = coins[i];
        promises.push(displayCoin(i, coin));
    }
    // wait for all promises to resolve
    await Promise.all(promises);
}

async function displayMove(i, move, age, civs, faces) {
    // skip moves from other players when extra moves are played by Babylon/Halikarnassos
    if (!move) {
        return;
    }
    // i-th player board wrap
    const boardWrap = document.getElementById(`player_board_${i}`);
    if (move.action === "BUILD") {
        // remove all "boarditem--active" from all elements within this boardwrap
        // boarditem--active used to highlight the last built card
        for (const item of boardWrap.querySelectorAll(".boarditem--active")){
            item.classList.remove("boarditem--active");
        }
        // create card wrap by card name
        const cardwrap = createCard(move.pick);
        const card = cardwrap.querySelector(".card");
        const cardname = cardwrap.querySelector(".cardname");
        let boarditemtype = CARDS[move.pick].boarditem;
        // find the boarditem docking zone by zonetype
        let zonetype = CARDS[move.pick].color;
        if (["brown", "grey"].includes(zonetype)){
            zonetype = "rsc";
        }
        let zone = boardWrap.querySelector(".zone--" + zonetype);
        // convert cardwrap to boarditem
        cardwrap.className = "boarditem";
        cardwrap.classList.add("boarditem--proto");
        cardwrap.classList.add("boarditem--active");
        boardWrap.appendChild(cardwrap);
        setTimeout(() => {
            cardwrap.removeChild(cardname);
            card.classList.add("card--type_" + boarditemtype);
            cardwrap.classList.remove("boarditem--proto");
            cardwrap.classList.add("boarditem--type_" + boarditemtype); 
            zone.appendChild(cardwrap);
        }, 500);
    } else if (move.action == "WONDER"){
        // create and center cardback in the middle of the board
        const cardback = createCardback(age);
        boardWrap.appendChild(cardback);
        // calculate wonder dock position
        let stage = boardWrap.querySelectorAll(".cardback").length;
        console.log("Wonder stage:", stage);
        // Handle special cases for Babylon and Gizah with Night side
        if (faces[i] === "Night" && civs[i] === "Babylon") {
            stage += 3;
        } else if (faces[i] === "Night" && civs[i] === "Gizah") {
            stage += 5;
        }
        // move cardback to wonder dock
        if (!cardback.classList.contains(`cardback--dock_${stage}`)) {
            cardback.classList.add(`cardback--dock_${stage}`);
        }
    }
}

async function displayCoin(i, coin) {
}