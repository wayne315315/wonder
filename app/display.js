import { createBoard, CIVS, FACES } from "./board.js";
import { displayHand, displayTrade, displayScore, removeCardFromHand } from "./hand.js";
import { createCard, createCardback, CARDS } from "./card.js";


// task.type : "SETTING", "FACE", "AGE", "MOVE", "UPDATE", "TRADE", "CLEAR", "BATTLE", "SCORE"
export async function display(task) {
    const banner = document.getElementById("banner");
    const hand = document.getElementById("hand");
    const info = document.getElementById("info");
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
            const users = task.users;
            
            const n = civs.length;
            for (let i = 0; i < n; i++){
                let civ = civs[i];
                let face =  (faces) ? faces[i] : null;
                if (face) {
                    createBoard(i, n, civ, face, users[i]);
                } else {                
                    createBoard(i, n, civ, "Night", users[i]);
                    let boardback = document.getElementById(`player_board_group_${i}`).firstChild;
                    boardback.classList.add("boardback");
                    createBoard(i, n, civ, "Day", users[i]);
                    // board group
                    let boardGroup = document.getElementById(`player_board_group_${i}`);
                    boardGroup.classList.add("board_group--face");
                }
            }
            // create player scoreboard TODO task.users
            if (!document.querySelector(".scoreboard")) {
                for (let i=0; i<n; i++){
                    // create scoreboard
                    let scoreboard = document.createElement("div");
                    scoreboard.id = `scoreboard_${i}`;
                    scoreboard.classList.add("scoreboard");
                    // create username
                    let username = document.createElement("div");
                    username.classList.add("user");
                    username.innerText = `${users[i]}`
                    // create battle zone
                    let battle = document.createElement("div");
                    battle.classList.add("battle");
                    // append username and battle zone to scoreboard
                    scoreboard.appendChild(username);
                    scoreboard.appendChild(battle);
                    // append scoreboard to info
                    info.appendChild(scoreboard);
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
            banner.innerText = "BATTLE";
            for (let i = 0; i < task.battle.length; i++) {
                const battlezone = document.querySelector(`#scoreboard_${i}`).querySelector(".battle");
                for (let result of task.battle[i]) {
                    if (result > 0){
                        let point = document.createElement("div");
                        point.classList.add(`win-${task.age}`);
                        battlezone.appendChild(point);
                    } else if (result < 0){
                        let point = document.createElement("div");
                        point.classList.add("lose");
                        battlezone.appendChild(point);
                    }
                }
            }
            break;
        case "SCORE":
            banner.innerText = "SCORE";
            displayScore(task);
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
        }, 100);
    } else if (move.action == "WONDER"){
        // create and center cardback in the middle of the board
        const cardback = createCardback(age);
        boardWrap.appendChild(cardback);
        // calculate wonder dock position
        let stage = boardWrap.querySelectorAll(".cardback").length;
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
    // i-th player bank
    const bank = document.querySelector(`#player_board_${i} .bank`);
    let coin_prev = parseInt(bank.innerHTML.match(/(\d+)$/)[0]);
    while (coin > coin_prev) {
        coin_prev++;
        // delay 10ms by setTimout with promise
        await new Promise(resolve => setTimeout(resolve, 10));
        bank.innerHTML = bank.innerHTML.replace(/\d+$/, coin_prev);
    }
    while (coin < coin_prev) {
        coin_prev--;
        await new Promise(resolve => setTimeout(resolve, 10));
        bank.innerHTML = bank.innerHTML.replace(/\d+$/, coin_prev);
    } 
}