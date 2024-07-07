import { Video } from 'expo-av';
import React from 'react';
import { View, Text, TouchableOpacity, StatusBar } from 'react-native'

const HomeScreen = ({ navigation }) => {
  return (
    <View className='flex-1 bg-white pt-10'>
      <StatusBar style='auto' />

        <View className='bg-white rounded-2xl
                          pt-8 pb-5 pl-5
                          w-90
                          border border-gray-200 border-2
                          mx-10'>
          <Text className='text-lg font-bold'>Hello,</Text>
          <Text className='text-lg font-black'>Lizzy Jamie</Text>
        </View>

      <Text className='text-xl font-black text-left ml-10 pt-9'>
        My videos
      </Text>

      <View className="flex flex-row mx-10">
        <View className="mr-8">
          <Video
              source={{ uri: "file:///data/user/0/host.exp.exponent/cache/ExperienceData/%2540anonymous%252Fsignify-f25c00f9-3b6f-4238-a980-cd671ffcf67b/ImagePicker/0bf2acbc-c88f-4153-a8f8-d9a9786a2021.mp4" }}
              rate={1.0}
              volume={1.0}
              isMuted={false}
              resizeMode='cover'
              shouldPlay={false}
              isLooping
              style={{ width: 150, height: 100}}
              className = "rounded-xl mt-5 mb-1"
              />
          <Text className="font-bold text-base">Today 🌥</Text>
          <Text className="text-xs">6 July 2024</Text>
        </View>

        <View className="">
          <Video
              source={{ uri: "file:///data/user/0/host.exp.exponent/cache/ExperienceData/%2540anonymous%252Fsignify-f25c00f9-3b6f-4238-a980-cd671ffcf67b/ImagePicker/437bd7bd-fc3a-4346-8222-b00c2bc6fca2.mp4" }}
              rate={1.0}
              volume={1.0}
              isMuted={false}
              resizeMode='cover'
              shouldPlay={false}
              isLooping
              style={{ width: 150, height: 100}}
              className = "rounded-xl mt-5 mb-1"
              />
          <Text className="font-bold text-base">Hello 👋</Text>
          <Text className="text-xs">6 July 2024</Text>
        </View>
      </View>

      <View className="flex flex-row mx-10">
        <View className="mr-8">
          <Video
              source={{ uri: "file:///data/user/0/host.exp.exponent/cache/ExperienceData/%2540anonymous%252Fsignify-f25c00f9-3b6f-4238-a980-cd671ffcf67b/ImagePicker/9a305df1-6d48-44c7-8fd3-b03fe8082716.mp4" }}
              rate={1.0}
              volume={1.0}
              isMuted={false}
              resizeMode='cover'
              shouldPlay={false}
              isLooping
              style={{ width: 150, height: 100}}
              className = "rounded-xl mt-5 mb-1"
              />
          <Text className="font-bold text-base">Thank you 🙏</Text>
          <Text className="text-xs">7 July 2024</Text>
        </View>

        <View className="">
          <Video
              source={{ uri: "file:///data/user/0/host.exp.exponent/cache/ExperienceData/%2540anonymous%252Fsignify-f25c00f9-3b6f-4238-a980-cd671ffcf67b/ImagePicker/4e0d26fd-bf39-469e-aa46-50c4ffd90e1a.mp4" }}
              rate={1.0}
              volume={1.0}
              isMuted={false}
              resizeMode='cover'
              shouldPlay={false}
              isLooping
              style={{ width: 150, height: 100}}
              className = "rounded-xl mt-5 mb-1"
              />
          <Text className="font-bold text-base">Hello All 😁</Text>
          <Text className="text-xs">7 July 2024</Text>
        </View>
      </View>
      
      
      <View className='bg-white rounded-lg
                          h-max
                          items-center'>
          {/* <Text className='text-lg font-bold mb-1.2 text-gray-300'>
            You haven't uploaded anything
          </Text> */}

          <View className='pt-10'>
            <TouchableOpacity
                onPress={() => navigation.navigate('Upload')}
                style={{
                  backgroundColor: '#32E08C',
                  paddingVertical: 16,
                  paddingHorizontal: 40,
                  borderRadius: 24,
                }}
              >
              <Text className='text-base text-white font-black'>
                Upload Video
              </Text>
            </TouchableOpacity>
          </View>
        </View>

    </View>
  );
};

export default HomeScreen;