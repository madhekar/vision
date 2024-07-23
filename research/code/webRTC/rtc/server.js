const express = require('express')

const io = require('socket.io')({
    path : '/webrtc',  // http://localhost:3001/webrtc/?EIO=3&transport=polling&t=NMp8rXl
})

const app = express()
const port = 8080

app.use(express.static(__dirname + '/build'))

app.get('/', (req, res, next) => {
    res.sendFile(__dirname + '/build/index.html')
})

const server =  app.listen(port, () => {
    console.log(`=> WebRTC App is listening on port ${port}`)
})

io.listen(server)

//let connectedPeers = new Map()

const webRTCNamespace = io.of('/webRTCPeers')

webRTCNamespace.on('connection', socket => {
    console.log('=> SocketID:',socket.id)

    socket.emit('connection-success', {
        status: 'connection-success',
        socketId: socket.id,
    })

    //connectedPeers.set(socket.id, socket)

    socket.on('disconnect', () => {
        console.log(`=> SocketID ${socket.id} has disconnected.`)
    })

    socket.on('sdp', data =>{
        console.log('=> SDP: ${data}')
        socket.broadcast.emit('sdp', data)
        //connectedPeers.delete(socket.id)
    })

    socket.on('candidate', data => {
        console.log('=> Candidate: ${data}')
        socket.broadcast.emit('candidate', data)
    })
})