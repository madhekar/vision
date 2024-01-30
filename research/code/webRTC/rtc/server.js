const express = require('express')

const io = require('socket.io')({
    path : '/webrtc'  // http://localhost:3001/webrtc/?EIO=3&transport=polling&t=NMp8rXl
})

const app = express()
const port = 8080

app.get('/', (req, res) => res.send('Hello, WebRTC!!'))

const server =  app.listen(port, () => {
    console.log(`WebRTC App is listening on port ${port}`)
})

io.listen(server)

const webRTCNamespace = io.of('/webRTCPeers')

webRTCNamespace.on('connection', socket => {
    console.log(socket.id)

    socket.emit('connection-sucess', {
        status: 'connection-sucess',
        socketId: socket.id,
    })

    socket.on('disconnect', () => {
        console.log(`A ${socket.id} has disconnected.`)
    })

    socket.on('sdp', data =>{
        console.log(data)
        socket.broadcast.emit('sdp', data)

    })

    socket.on('candidate', data => {
        console.log(data)
        socket.broadcast.emit('candidate', data)
    })
})