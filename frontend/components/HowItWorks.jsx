import React, { useState } from 'react';
import { View, Text, Image, TouchableOpacity } from 'react-native';
import Modal from 'react-native-modal';

const Header = () => {
  const [isModalVisible, setModalVisible] = useState(false);

  const toggleModal = () => {
    // console.log("Toggle Modal called:", !isModalVisible); // Debug log
    setModalVisible(!isModalVisible);
  };

  return (
    <View>

      <TouchableOpacity 
        onPress={toggleModal} 
        style={{
          width: 40,  // Equal width and height
          height: 40, // Equal width and height
          borderRadius: 20, // Half of the width/height
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <Image 
              source={require('../assets/question_small.png')} 
              className="w-10 h-10" 
              resizeMode="contain" 
            />
      </TouchableOpacity>

      <Modal 
        isVisible={isModalVisible} 
        onBackdropPress={toggleModal}
        useNativeDriver
        hideModalContentWhileAnimating
        backdropTransitionOutTiming={0}
      >
        <View className="bg-white p-6 rounded-3xl"> 
          <View className="flex flex-row items-center justify-start m-2"> 
            <Image 
              source={require('../assets/icon_green.png')} 
              className="w-10 h-10" 
              resizeMode="contain" 
            />
            <Text className="font-bold text-xl ml-2">How does Signify work?</Text> 
          </View>
          
          <Text className="text-sm m-2">Signify provides auto-generated captions for your videos by using Generative AI.
          Upload a maximum of 1-minute video and Signify will generate the captions automatically for you!</Text>
          <TouchableOpacity onPress={toggleModal}>
            <Text className="text-xs font-bold mt-5 text-center">Tap anywhere to return</Text>
          </TouchableOpacity>
        </View>
      </Modal>
    </View>
  );
};

export default Header;
