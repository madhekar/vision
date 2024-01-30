
import './App.css';
import React, {useRef, useEffect} from 'react';
import io from 'socket.io-client'

const socket = io(
  '/webRTCPeers',
  { 
    path: '/webrtc'
  }
)

function App() {
  const localVideoRef = useRef()
  const remoteVideoRef = useRef()
  const pc = useRef(new RTCPeerConnection(null))
  const textRef = useRef()
  const candidates = useRef([])

  useEffect(() =>{

    socket.on('connection-success', success => {
      console.log(success)
    })

    socket.on('sdp', data => {
      console.log(data)
      textRef.current.value = JSON.stringify(data.sdp)
    })

    socket.on('candidate', candidate => {
      console.log(candidate)
      candidates.current = [...candidates.current, candidate]
    })

      const constraints = {
        audio : false,
        video : true,
      }

   navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        // display stream
        localVideoRef.current.srcObject = stream

        stream.getTracks().forEach(track => {
          _pc.addTrack(track, stream)
        })
      })
      .catch(e => {
        console.log('getUserMedia error ...', e)
      }) 

      const _pc =  new RTCPeerConnection(null)
      _pc.onicecandidate = (e) => {
        if (e.candidate)
          console.log(JSON.stringify(e.candidate))
          socket.emit('candidate', e.candidate)
      }

      // connected, disconnected, failed, closed 
      _pc.oniceconnectionstatechange = (e) => {
        console.log(e)
      }

      _pc.ontrack = (e) =>{
        // got remote stream
        remoteVideoRef.current.srcObject = e.streams[0]
      }
      pc.current = _pc
  }, [])

  const createOffer = () => {
    pc.current.createOffer({
      offerToReceiveAudio: 1,
      offerToReceiveVidio: 1,
    }).then(sdp => {
      console.log(JSON.stringify(sdp))
      pc.current.setLocalDescription(sdp)

      // send sdp to the server
      socket.emit('sdp', {
        sdp,
      })
    }).catch(e => console.log(e))
  }

  const createAnswer = () => {
    pc.current.createAnswer({
      offerToReceiveAudio: 1,
      offerToReceiveVidio: 1,
    }).then(sdp => {
      console.log(JSON.stringify(sdp))
      pc.current.setLocalDescription(sdp)

      //send the answer sdp to the offering peer
      socket.emit('sdp', {
        sdp,
      })
    }).catch(e => console.log(e))
  }

  const setRemoteDescription = () => {
    // get SDP value from the text editor
    const sdp = JSON.parse(textRef.current.value)
    console.log(sdp)

    pc.current.setRemoteDescription(new RTCSessionDescription(sdp))
  }

  const addCandidate = () => {
    //const candidate =  JSON.parse(textRef.current.value)
    //console.log('Additing Candidate...', candidate)

    candidates.current.forEach(candidate => {
      console.log(candidate)
      pc.current.addIceCandidate(new RTCIceCandidate(candidate))
    })
  }


  return (
    <div style={{margin:10}}>
      <video style={{
        width: 240, height: 240,
        margin: 5, backgroundColor: 'black',
      }}
      ref={localVideoRef} autoPlay></video>

      <video style={{
        width: 240, height: 240,
        margin: 5, backgroundColor: 'black',
      }}
      ref={remoteVideoRef} autoPlay></video>

      <br />
      <button onClick={createOffer}>Offer</button>
      <button onClick={createAnswer}>Answer</button>
      <br />
      <textarea ref={textRef}></textarea>
      <br />
      <button onClick={setRemoteDescription}>Remote Description</button>
      <button onClick={addCandidate}>Add Candidates</button>
    </div>
  );
}

export default App;
