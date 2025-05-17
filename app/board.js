export const CIVS = [
    "Gizah",
    "Babylon",
    "Olympia",
    "Rhódos",
    "Éphesos",
    "Alexandria",
    "Halikarnassos",
];

export const FACES = ["Day", "Night"];

export function createBoard(i, n, civ, face, username) {
    const bg_unit_y = 100 / 13;
    const bg_pos_y = (7 * (face === "Night") + CIVS.indexOf(civ)) * bg_unit_y;
    const LEFT = 23;
    const RIGHT = 67;
    const MID = 45;
    const TOP = 20;
    const INTERVAL = 30;
    const BOTTOM = 15;
    // create board
    let top = TOP;
    let left = MID;
    let height = top;
    const pos = [{ top: top, left: left }];
    for (let j = 1; j < n; j++) {
        if (2 * j < n) {
            top += INTERVAL;
            left = RIGHT;
        } else if (2 * j === n) {
            top += INTERVAL;
            left = MID;
        } else if (2 * j === n + 1) {
            left = LEFT;
        } else {
            top -= INTERVAL;
            left = LEFT;
        }
        pos.push({ top: top, left: left });
        height = Math.max(height, top);
    }
    // create board
    const board = document.createElement("div");
    board.classList.add("board");
    board.style.backgroundPositionY = `${bg_pos_y}%`;
    // create board group
    const boardGroupID = `player_board_group_${i}`;
    let boardGroup = document.getElementById(boardGroupID);
    if (!boardGroup) {
        boardGroup = document.createElement("div");
        boardGroup.id = boardGroupID;
        boardGroup.classList.add("board_group");
    }
    // only create board wrap if it doesn't exist
    const boardID = `player_board_${i}`;
    let boardWrap = document.getElementById(boardID);
    if (!boardWrap) {
        boardWrap = document.createElement("div");
        boardWrap.id = boardID;
        boardWrap.classList.add("board_wrap");
        boardWrap.style.top = `${pos[i].top}vw`;
        boardWrap.style.left = `${pos[i].left}vw`;
    }
    // create zone
    const zones = []
    const zonetypes = ["rsc", "red", "blue", "yellow", "green", "purple", "buffer"];
    for (const zonetype of zonetypes) {
        let zone = document.createElement("div");
        zone.classList.add("zone");
        zone.classList.add(`zone--${zonetype}`);
        zones.push(zone);
    }
    // create player wrap
    const playerWrap = document.createElement("div");
    playerWrap.classList.add("whiteblock");
    // create bank
    const bank = document.createElement("div");
    bank.classList.add("bank");
    bank.innerHTML = `<span class="coin"></span>&nbsp;&nbsp;3`;
    // create user
    const user = document.createElement("div");
    user.classList.add("user");
    user.classList.add("user--whiteblock");
    user.innerText = `${username}`;
    // user append to player wrap
    playerWrap.appendChild(user);
    // bank append to player wrap
    playerWrap.appendChild(bank);
    // board append to board group
    boardGroup.appendChild(board);
    // board group append to board wrap
    boardWrap.appendChild(boardGroup);
    // player wrap append to board wrap
    boardWrap.appendChild(playerWrap);
    // zone append to board wrap TODO
    for (const zone of zones) {
        boardWrap.appendChild(zone);
    }
    // board wrap append to arena
    const arena = document.querySelector("#arena");
    arena.appendChild(boardWrap);
    // set arena height
    height += BOTTOM;
    arena.style.height = `${height}vw`;
    // set main min-height
    const main = document.querySelector("main");
    main.style.minHeight = `${100}vh`;
}
