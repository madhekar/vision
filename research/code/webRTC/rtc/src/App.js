
import './App.css';
import React, {useRef, useEffect, useState} from 'react';
import io from 'socket.io-client'

const socket = io(
  '/webRTCPeers',
  { 
    path: '/webrtc'
  }
)

function App() {
  const pc_config = null
  const pc_config_new ={
      'iceServers' : [
        {
            'urls': 'stun:[STUN-IP]:[PORT]',
            'credential' : '[CREDENTIAL]',
            'username' : '[USERNAME]'
        }
      ]
  }
  const localVideoRef = useRef()
  const remoteVideoRef = useRef()
  const pc = useRef(new RTCPeerConnection(pc_config))
  const textRef = useRef()

  const [offerVisible, setOfferVisible] = useState(true)
  const [answerVisible, setAnswerVisible] = useState(false)
  const [status, setStatus] = useState('Make a call now')

  useEffect(() =>{

    socket.on('connection-success', success => {
      console.log(success)
    })

    // SDP session
    socket.on('sdp', data => {
      console.log(data)
      pc.current.setRemoteDescription(new RTCSessionDescription(data.sdp))
      textRef.current.value = JSON.stringify(data.sdp)

      if(data.sdp.type === 'offer'){
        setOfferVisible(false)
        setAnswerVisible(true)
        setStatus('Incomming call...')
      }else {
        setStatus('Call established.')
      }
    })

    // ICE candidate
     socket.on('candidate', candidate => {
      console.log(candidate)
      pc.current.addIceCandidate(new RTCIceCandidate(candidate))
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
        if (e.candidate){
          console.log(JSON.stringify(e.candidate))
          sendToPeer('candidate', e.candidate)
        }
      }

      // connected, disconnected, failed, closed 
      _pc.oniceconnectionstatechange = (e) => {
        console.log(e)
      }

      _pc.ontrack = (e) => {
        // got remote stream
        remoteVideoRef.current.srcObject = e.streams[0]
      }

      pc.current = _pc

  }, [])

  const sendToPeer = (eventType, payload) => {
    socket.emit(eventType, payload)
  }

  const processSDP = (sdp) => {
     console.log(JSON.stringify(sdp))
     pc.current.setLocalDescription(sdp)
     sendToPeer( 'sdp', { sdp })
    }

  const createOffer = () => {
    pc.current.createOffer({
      offerToReceiveAudio: 1,
      offerToReceiveVideo: 1,
    }).then(sdp => {
      processSDP(sdp)
      setOfferVisible(false)
      setStatus('Calling...')
    }).catch(e => console.log(e))
  }

  const createAnswer = () => {
    pc.current.createAnswer({
      offerToReceiveAudio: 1,
      offerToReceiveVideo: 1,
    }).then(sdp => {
      //send the answer sdp to the offering peer
      processSDP(sdp)
      setAnswerVisible(false)
      setStatus('Call established.')
    }).catch(e => console.log(e))
  }

  const showHideButton = () => {
    if (offerVisible){
      return (
        <div>
          <button onClick={createOffer}>Call</button>
        </div>
      ) 
    } else if (answerVisible){
      return(
        <div>
          <button onClick={createAnswer}>Answer</button>
        </div>
      )
    }
  }

  return (
    <div style={{margin:10}}>
      <video style={{
        width: 240, height: 240, margin: 5, backgroundColor: 'black',
      }}
      ref={localVideoRef} autoPlay></video>

      <video style={{
        width: 240, height: 240, margin: 5, backgroundColor: 'black',
      }}
      ref={remoteVideoRef} autoPlay></video>

      <br />
        {showHideButton()}
        <div>{status}</div>
      <br />
        <textarea ref={textRef}></textarea>
    </div>
  );
}

export default App;
