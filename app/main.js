import { io } from "https://cdn.socket.io/4.8.1/socket.io.esm.min.js";
import { display } from "./display.js";


const URL = "http://127.0.0.1:5000";
const FACES = ["Day", "Night"];
const ACTIONS = ["BUILD", "WONDER", "DISCARD"];

let age = 1;
let civs = null;
let faces = null;
let gid = null;
let users = null;

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
    const payload = {"uid": uid, "players": ["H", "R", "R", "R", "R", "R", "R"], "random_face": false};
    console.log("sending create");
    socket.emit("create", payload);
    console.log("create sent");
    console.log("waiting for game");
    return new Promise((resolve) => {
        const gameListener = (data) => {
            console.log("game created");
            socket.off("game", gameListener);
            socket.off("cancel_create", cancelListener);
            resolve(data);
        };
        const cancelListener = () => {
            console.log("game creation cancelled");
            socket.off("game", gameListener);
            socket.off("cancel_create", cancelListener);
            resolve(undefined);
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

async function waitForDetail(eventName, elements) {
    return new Promise((resolve) => {
        // define handler
        const handler = (event) => {
            if (event.detail !== undefined) {
                // remove the listener from all elements
                elements.forEach(element => {
                    element.removeEventListener(eventName, handler);
                });
                // resolve the promise with the event detail
                resolve(event.detail);
            }
        };
        // add event listener to each element
         elements.forEach(element => {
            element.addEventListener(eventName, handler);
        });
    })
}

async function process_task(socket, uid, task) {
    console.log("task : ", task);
    // inject users, gid, age, civs, faces if it doesn't exist in task
    let res;
    if (!task.age) {
        task.age = age;
    }
    if (!task.civs) {
        task.civs = civs;
    }
    if (!task.faces) {
        task.faces = faces;
    }
    if (!task.gid) {
        task.gid = gid;
    }
    if (!task.users) {
        task.users = users;
    }
    // display task
    await display(task);
    // process task
    if (task.type === "GAME") {
        await create_game(socket, uid);
    } else if (task.type === "SETTING"){
        civs = task.civs;
        faces = task.faces;
        gid = task.gid;
        users = task.users;
    } else if (task.type === "AGE"){
        age = task.age;
    } else if (task.type === "FACE"){
        let facebuttons = document.querySelectorAll(".facebutton");
        res = await waitForDetail("face",  facebuttons);
        console.log("face : ", res);
    } else if (task.type === "MOVE") {
        if (task.asked) {
            const banner = document.getElementById("banner");
            banner.innerText = "** Previous move is invalid. Retry...";
            const color_prev = banner.style.color;
            banner.style.color = "red";
            await new Promise(resolve => setTimeout(resolve, 1));
            banner.style.color = color_prev; 
        }
        res = await waitForDetail("move", document.querySelectorAll("#hand .cardmenu"));
    } else if (task.type === "TRADE") {
        let trades = document.querySelectorAll(".trade");
        res = await waitForDetail("trade",  trades);
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
    // old display tasks
    let endpoint = `${URL}/done`;
    const tasks_done = await fetch(endpoint).then((response) => {
        return response.json()
    })
    console.log("tasks_done:",  tasks_done)
    for (let task of tasks_done) {
        await process_task(socket, uid, task);
    }
    // active task right before reloading
    endpoint = `${URL}/current`;
    const tasks_current = await fetch(endpoint).then((response) => {
        return response.json()
    })
    console.log("tasks_current:",  tasks_current)
    for (let task of tasks_current) {
        let res = await process_task(socket, uid, task);
        submit(res);
    }
    // new tasks
    while (true){
        let task = await fetch_task();
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
