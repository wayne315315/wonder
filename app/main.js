import { io } from "https://cdn.socket.io/4.8.1/socket.io.esm.min.js";

const URL = "http://127.0.0.1:5000";
const FACES = ["Day", "Night"];
const ACTIONS = ["BUILD", "WONDER", "DISCARD"];


function getCookie(name) {
    const nameEQ = name + "=";
    const ca = document.cookie.split(';');
    for (let i = 0; i < ca.length; i++) {
      let c = ca[i];
      while (c.charAt(0) === ' ') c = c.substring(1, c.length);
      if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
    }
    return null;
  }

async function getUID(){
    const uid = await fetch(URL)
        .then((response) => {
            if (response.ok) {
                return getCookie("uid");
            }
            else {
                throw new Error("getUID() HTTP GET request failed");
            }
        })
    console.log("uid : ", uid);
    return uid;
}

async function fetch_task() {
    const endpoint = `${URL}/dequeue`;
    console.log(`endpoint : ${endpoint}`);
    let task = fetch(endpoint, {
        method: "GET",
        credentials: "include",
        }).then((response) => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error("fetch_task() HTTP GET request failed");
            }
        }).catch((error) => {
            throw new Error("Connection failed during fetch_task()...");
        });
    return task;
}

async function create_game(socket, uid) {
    const payload = {"uid": uid, "players": ["H", "H", "R"], "random_face": false};
    console.log("sending create");
    socket.emit("create", payload);
    console.log("create sent");
    console.log("waiting for game");
    return new Promise((resolve, reject) => {
        const gameListener = (data) => {
            console.log("game created");
            socket.off("game", gameListener);
            socket.off("cancel_create", cancelListener);
            resolve(data.gid);
        };
    
        const cancelListener = () => {
            console.log("game creation cancelled");
            socket.off("game", gameListener);
            socket.off("cancel_create", cancelListener);
            reject(undefined);
        };
    
        socket.once("game", gameListener);
        socket.once("cancel_create", cancelListener);
    });
}

async function join_game(socket, uid) {
    const payload = {"uid": uid};
    console.log("sending join");
    socket.emit("join", payload);
    console.log("join sent");
    console.log("waiting for game");
    return new Promise((resolve, reject) => {
        const gameListener = (data) => {
            console.log("game joined");
            socket.off("game", gameListener);
            socket.off("cancel_join", cancelListener);
            resolve(data.gid);
        };
    
        const cancelListener = () => {
            console.log("game joining cancelled");
            socket.off("game", gameListener);
            socket.off("cancel_join", cancelListener);
            reject(undefined);
        };
    
        socket.once("game", gameListener);
        socket.once("cancel_join", cancelListener);
    });
}


async function process_task(socket, uid, task) {
    let res;
    if (task.type === "GAME") {
        let gid = await create_game(socket, uid);
    } else if (task.type === "FACE"){
        const face = FACES[Math.floor(Math.random() * FACES.length)];
        res = {"face": face};
    } else if (task.type === "MOVE") {
        const hand = task.hand;
        const asked = task.asked;
        const pick = hand[Math.floor(Math.random() * hand.length)];
        const action = ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
        res = {"pick": pick, "action": action};
        if (asked) {
            console.log("** Previous move is invalid. Retry...");
        }
    } else if (task.type === "TRADE") {
        const coins = task.coins;
        const trade = coins[0];
        res = {"trade": trade};
    } else {
        console.log(task);
    }
    return res;
}

function submit(res){
    const endpoint = `${URL}/submit`;
    fetch(endpoint, {
        method: "POST",
        credentials: "include",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(res)
    }).then((response) => {
        if (response.ok) {
            console.log("Result submitted");
        } else {
            throw new Error("submit() HTTP POST request failed");
        }
    }).catch((error) => {
        throw new Error("Connection failed during submit()...");
    });
}



export async function main(){
    const uid = await getUID();
    const socket = io(URL);
    while (true){
        let task = await fetch_task();
        console.log("task : ", task);
        let res = await process_task(socket, uid, task);
        if (res){
            submit(res);
        }
        if (task.type === "SCORE") {
            console.log("Game finished");
            break;
        }
    }
}

main();
