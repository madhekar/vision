
import './App.css';
import React, {useRef, useEffect, useState} from 'react';
import io from 'socket.io-client'
import Video from './Components/Video';


const socket = io(
  '/webRTCPeers',
  { 
    path: '/webrtc',
    query: {}
  }
)

function App() {
   const pc_config ={
      'iceServers' : [
        { urls : 'stun:stun.l.google.com:19302'},
        { urls : 'stun:stun2.l.google.com:19302'}
      ]
  } 

  //const localVideoRef = useRef()
  const remoteVideoRef = useRef()
  const pc = useRef(new RTCPeerConnection(pc_config))
  const textRef = useRef()

  const [offerVisible, setOfferVisible] = useState(true)
  const [answerVisible, setAnswerVisible] = useState(false)
  const [status, setStatus] = useState('Make a call now')
  const [localStream, setLocalStream] = useState(null)

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
        audio : true,
        video : true,
        options: {
          mirror: true,
        }
      }

     //const _pc =  new RTCPeerConnection(null)

   navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        // display local stream
        //localVideoRef.current.srcObject = stream
        setLocalStream(stream)
        //window.localStream = stream
        //_pc.addStream(stream)
       // setLocalStream(stream)
    
        stream.getTracks().forEach(track => {
          _pc.addTrack(track, stream)
        })
      })
      .catch(e => {
        console.log('getUserMedia error occred ...', e)
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
        console.log('inside on ice connection stete change ',e)
      }

      _pc.ontrack = (e) => {
        // got the remote stream
        remoteVideoRef.current.srcObject = e.streams[0]
      }
      pc.current = _pc
  }, [])  // end of useEffect method

  const sendToPeer = (eventType, payload) => {
    socket.emit(eventType, payload)
  }

  const processSDP = (sdp) => {
     console.log(JSON.stringify(sdp))
     pc.current.setLocalDescription(sdp)
     sendToPeer( 'sdp', { sdp })
    }

  // create offer  
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

  // create answer
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
          <button onClick={createOffer} style={{width: 50, height:40, margin: 5}}>Call</button>
               ( {status} ) 
        </div>
      ) 
    } else if (answerVisible){
      return(
        <div>
          <button onClick={createAnswer} style={{width: 80, height:40, margin: 5}}>Answer</button>
          ( {status} )
        </div>
      )
    }
  }
 //console.log(this.state.localStream)
  return (

    <div style={{margin:5, position: 'absolute'}}> 
     <div style={{zIndex:4, position:'fixed'}}>
      <br/>
        {showHideButton()}
         {/* <div>{status}</div>  */}
         <textarea ref={textRef} hidden= {false} style={{margin: 5}}></textarea>  
      <br />
      </div>

{/* local video stream */}
      <Video 
      videoStyles={{
        position: 'fixed', 
        right: 0, 
        width: 300,   
        margin: 5, 
        resizeMode: 'stretch',
        backgroundColor: 'black',  
        zIndex: 2
      }}
      //ref={localVideoRef} 
      videoStream = {this.state.localStream}
      autoPlay/>


{/* remote video stream */}
 {/*      <video  style={{
        position: 'fixed', 
        bottom: 0, 
        minWidth: '100%', 
        minHeight:'100%', 
        margin: 5, 
        backgroundColor: 'black', 
        zIndex: 1
      }}
      ref={remoteVideoRef} 
      autoPlay /> */}
     </div>
  );
 }


export default App;
