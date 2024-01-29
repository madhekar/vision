import logo from './logo.svg';
import './App.css';
import {useRef, useEffect} from 'react';

function App() {
  const localVideoRef = useRef()
  const remoteVideoRef = useRef()

  const getUserMedia = async () => {
    const constraints = {
      audio : false,
      video : true,
    }
 /*    navigator.mediaDevices.getUserMedia(constraints)
    .then(stream => {
      // display stream
      localVideoRef.current.srcObject = stream
    })
    .catch(e => {
      console.log('getUserMedia error ...', e)
    }) */
    const stream = await navigator.mediaDevices.getUserMedia(constraints)
    localVideoRef.current.srcObject = stream
  }
  return (
    <div style={{margin:10}}>
      <button onClick={() => getUserMedia()}>Get Media Access</button>
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
      <button onClick={() => {}}>Offer</button>
      <button onClick={() => {}}>Answer</button>
      <br />
      <button onClick={() => {}}>Remote Description</button>
      <button onClick={() => {}}>Add Candidates</button>
    </div>
  );
}

export default App;
