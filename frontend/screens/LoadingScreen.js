import React, { useEffect, useState } from 'react';
import { View, Text, ImageBackground, Image, Modal} from 'react-native';


const LoadingScreen = () => {

  return (
    <Modal
        isVisible={true} 
        useNativeDriver
        hideModalContentWhileAnimating
        backdropTransitionOutTiming={0}>
        <ImageBackground
            source={require('../assets/background.png')} 
            style={{ width: '100%', height: '100%' }} 
            resizeMode="cover" 
        >
        <View className="justify-center items-center mt-40 pt-20">
            <Image 
              source={require('../assets/icon.png')}
              style={{ width: 150, height: 150 }} 
              resizeMode="contain" 
            />
            <Image 
              source={require('../assets/loading.gif')}
              style={{ width: 50, height: 50 }} 
              resizeMode="contain" 
              className = "my-10"
            />

            <Text className="text-lg font-bold text-center">Loading...</Text>

        </View>
            

        </ImageBackground>

    </Modal>
        
  );
};

export default LoadingScreen;