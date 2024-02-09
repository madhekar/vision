/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 * @flow strict-local
 */

import React from 'react';

import {
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  Dimensions,
  TouchableOpacity,
  ScrollView,
  StatusBar
} from 'react-native';

import{
  RTCPeerConnection,
  RTCIceCandidate,
  RTCSessionDescription,
  RTCView,
  MediaStream,
  MediaStreamTrack,
  mediaDevices,
  registerGlobals
} from 'react-native-webrtc'

import io from 'socket.io-client'

const dimention = Dimensions.get('window')

class App extends React.Component {

  constructor(props){
    super(props)
    this.state = {
      localStream: null,
      remoteStream: null,
    }
    this.pc
    this.sdp 
    this.socket = null
    this.candidates = []
  }

  componentDidMount = () => {

    this.socket = io.connect(
      'https://c3af-70-137-105-159.ngrok-free.app/webRTCPeers',
      {
        path: '/webrtc',
        query: {}
      }
    )

    this.socket.on('connection-success', success => {
      console('***URL: ', this.socket.toURL())
      console.log(success)
    })

    this.socket.on('sdp', data => {

      this.sdp = JSON.stringify(data.sdp)
      this.pc.setRemoteDescription(new RTCSessionDescription(data.sdp))
      if(data.sdp.type === 'offer'){
        console.log("Incomming Call...")
      }else{
        console.log("Call Established.")
      }
    })

 /*    this.socket.on('answer', (sdp) => {
      this.sdp = JSON.stringify(sdp)

      this.pc.setRemoteDescription(new RTCSessionDescription(sdp))
    }) */

    this.socket.on('candidate', (candidate) =>{
      this.candidates = [...this.candidates, candidate]
      this.pc.addIceCandidate(new RTCIceCandidate(candidate))
    })

    const pc_config = {
      'iceServers' : [
        {
          urls: 'stun:stun.l.google.com:19302'
        }
      ]
    }

    this.pc = new RTCPeerConnection()

    this.pc.setConfiguration(pc_config)

    this.pc.getStats().then(desc => console.log('PeerConnection stats:',desc))

    this.pc.onicecandidate = (e) => {
      if(e.candidate) {
        console.log(JSON.stringify(e.candidate))
        this.sendToPeer('candidate', e.candidate)
      }
    }

    this.pc.oniceconnectionstatechange = (e) => {
      console.log(e)
    }

    this.pc.ontrack = (e) => {
      // got remote stream
      debugger

      console.log('***on track')
      this.setState({
        remoteStream : e.stream
      })
    }

  /*    this.pc.onaddstream = (e) => {
      debugger
      console.log('***on add stream')
      this.setState({
        remoteStream : e.stream
      })
    } */ 
    
    const works = (stream) =>{
      console.log('***streamURL:', stream.toURL())
      this.setState({
        localStream : stream
      })
      //this.pc.addStream(stream)
      stream.getTracks().forEach(track => this.pc.addTrack(track, stream))
    }

    const fails = (e) => {
      console.log('getUserMedia Error: ', e)
    }

    let isFront = true;
    mediaDevices.enumerateDevices().then(sourceInfos => {
      console.log('sourceInfos:', sourceInfos);
      let videoSourceId;
      for (let i =0 ; i < sourceInfos.letgth; i++) {
        const sourceInfo = sourceInfos[i];
        if(sourceInfo.kind == "videoinput" && sourceInfo.facing == (isFront ? "front" : "environment")) {
          videoSourceId = sourceInfo.deviceId
        }
      }

      const constraints = {
        audio: true,
        video: {
          mandatory: {
            minWidth: 500,
            minHeight: 300,
            minFrameRate: 30
        },
        facingMode: (isFront ? "user" : "environment"),
        optional: (videoSourceId ? [{ sourceId: videoSourceId}] : [])
      }
    }
      const constraints_ ={audio: true, video:true}
      console.debug(constraints)

      mediaDevices.getUserMedia(constraints).then(works).catch(fails)
    });
  }

/*   sendToPeer = (messageType, payload) => {
    console.log('****sendToPeer', messageType)
    this.socket.emit(messageType, {
      socketID : this.socket.id,
      payload
    })
  }  */
    

   sendToPeer = (eventType, payload) => {
    this.socket.emit(eventType, payload)
  }

   processSDP = (sdp) => {
     console.log(JSON.stringify(sdp))
     this.pc.setLocalDescription(sdp)
     this.sendToPeer( 'sdp', { sdp })
    }

  createOffer = () => {
    console.log('***offer***')

    this.pc.createOffer({
      offerToReceiveAudio: 1,
      offerToReceiveVideo: 1,
    }).then(sdp =>{
      this.processSDP(sdp)
      // console.log(JSON.stringify(sdp))

      // this.pc.setLocalDescription(sdp)

      // this.sendToPeer('sdp', sdp)
    }).catch(e=>console.debug('error creating offer', e))
  }

  createAnswer = () => {
    console.log('***answer***')

    this.pc.createAnswer({ 
      offerToReceiveAudio: 1,
      offerToReceiveVideo: 1,
    }).then(sdp => {
      this.processSDP(sdp)
      // console.log(JSON.stringify(sdp))

      // this.pc.setLocalDescription(sdp)

      // this.sendToPeer('sdp', sdp)
    }).catch(e => console.debug('error creating answer', e))
  }

/*   setRemoteDescription = () => {

    const desc = JSON.parse(this.sdp)

    this.pc.setRemoteDescription(new RTCSessionDescription(desc))
  } */

  addCandidate = () => {

    this.candidates.forEach(candidate => {
      console.log(JSON.stringify(candidate))
      this.pc.addIceCandidate(new RTCIceCandidate(candidate))
    });
  }
  

 render(){

  const {
    localStream,
    remoteStream
  } = this.state

  const remoteVideo = remoteStream ?
  (
    <RTCView
    key = {2}
    mirror = {true}
    style = {{...styles.rtcViewRemote}}
    objectFit='contain'
    streamURL={remoteStream && remoteStream.toURL()}
    />

  ) :
  (
    <View style={{ padding : 15,}}>
      <Text style={{fontSize:22, textAlign: 'center', color: 'white'}}>waiting for the peer connection...</Text>
    </View>
  )

  return (

    <SafeAreaView style={{flex: 1,}}>
      <StatusBar backgroundColor='blue' barStyle={'dark-content'}/>
      <View style={{...styles.buttonsContainer}}>
        <View style={{flex: 1, }}>
          <TouchableOpacity onPress={this.createOffer}> 
          <View style={styles.button}>
            <Text style={{...styles.textContent, }}>Call</Text>
          </View >
          </TouchableOpacity>
        </View>
        <View style={{flex: 1,}}>
          <TouchableOpacity onPress={this.createAnswer}> 
          <View style={styles.button}>
            <Text style={{...styles.textContent, }}>Answer</Text>
          </View>
          </TouchableOpacity>
        </View>
      </View>
      <View style={{...styles.videosContainer, }}>
      <View style= {{
        position: 'absolute',
        zIndex: 1000,
        bottom: 10,
        right: 10,
        width: 100, 
        height: 200,
        backgroundColor: 'black',
      }}>
        <View style={{flex: 1,}}>
          <TouchableOpacity onPress={() => localStream._tracks[1]._switchCamera()}>
            <View>
              <RTCView 
              key={1}
              zOrder={0}
              objectFit='cover'
              style={{...styles.rtcView}}
              streamURL={localStream && localStream.toURL()}
              /> 
            </View>
          </TouchableOpacity>
        </View>
      </View>  
      <ScrollView style={{ ...styles.scrollView}}>
        <View style={{
          flex: 1,
          width:'100%',
          backgroundColor: 'black',
          justifyContent:'center',
          alignItems: 'center',
        }}>
       { remoteVideo }
       </View>
      </ScrollView>
      </View>
    </SafeAreaView>
  );
 }
};

const styles = StyleSheet.create({
  buttonsContainer: {
    flexDirection: 'row',
  },
  button: {
    margin: 5,
    paddingVertical: 10,
    backgroundColor: 'lightgray',
    borderRadius: 5,
  },
  textContent: {
    fontFamily: 'Avenir',
    fontSize: 20,
    textAlign: 'center',
  },
  videosContainer: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'center',
  },
  rtcView: {
    width: 100,
    height: 200,
    backgroundColor: 'black',
  },
  scrollView: {
    flex: 1,
    backgroundColor: 'teal',
    padding: 15,
  },
  rtcViewRemote: {
    width: Dimensions.width - 30,
    backgroundColor: 'black',
    height: 200,
  }

});

export default App;
