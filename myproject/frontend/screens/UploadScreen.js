import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Alert } from 'react-native'
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';

const UploadScreen = ({ navigation }) => {
    const [video, setVideo] = useState(null);
  
    const handleChooseVideo = async () => {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: false,
        quality: 1,
      });
  
      console.log('ImagePicker result:', result);

      if (!result.canceled) {
        const { uri } = result.assets[0];
        setVideo(uri);
        await uploadVideo(uri);
        navigation.navigate('Generated', { videoUri: uri });
      }
    };

    const uploadVideo = async (uri) => {
      const formData = new FormData();
      formData.append('video', {
        uri,
        type: 'video/mp4',
        name: 'upload.mp4',
      });
  
      try {
        const response = await axios.post('http://192.168.1.104:8000/api/upload/', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        Alert.alert('Upload Success', JSON.stringify(response.data));
      } catch (error) {
        Alert.alert('Upload Failed', error.message);
      }
    };

    

  return (
    <View className='flex-1 items-center bg-white justify-center'>
        <TouchableOpacity
            onPress={handleChooseVideo} 
            style={{
                backgroundColor: '#32E08C',
                paddingVertical: 16,
                paddingHorizontal: 40,
                borderRadius: 24,
            }}
            >
            <Text className='text-base text-white font-black'>
                Upload from Gallery
            </Text>
        </TouchableOpacity>
    </View>
  );
};

export default UploadScreen;