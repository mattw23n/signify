import React from 'react';
import { View, Text, TouchableOpacity, StatusBar } from 'react-native'

const HomeScreen = ({ navigation }) => {
  return (
    <View className='flex-1 bg-white pt-10'>
      <StatusBar style='auto' />

        <View className='bg-white rounded-lg
                          pt-12 pb-5 pl-5
                          w-72
                          border border-gray-200
                          mx-auto'>
          <Text className='text-lg font-bold mb-1.2'>Hello,</Text>
          <Text className='text-lg font-black'>Lizzy Jamie</Text>
        </View>

      <Text className='text-lg font-black text-left pl-10 pt-9'>
        My videos
      </Text>
      
      <View className='bg-white rounded-lg
                          pt-36 h-max
                          items-center'>
          <Text className='text-lg font-bold mb-1.2 text-gray-300'>
            You haven't uploaded anything
          </Text>

          <View className='pt-36'>
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