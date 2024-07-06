import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Ionicons from 'react-native-vector-icons/Ionicons';
import HomeScreen from "../screens/HomeScreen";
import UploadScreen from "../screens/UploadScreen";

const Tab = createBottomTabNavigator();

export default function NaviBar() {
  return (
    <Tab.Navigator
        screenOptions={{
            headerShown: false,
            tabBarActiveTintColor: "#32E08C",
            tabBarInactiveTintColor: "#A5A5A5",
            tabBarLabelStyle: {
                fontSize: 12.5,
                fontWeight: 900
            },
        }}>
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{
            tabBarIcon: ({color, size }) => (
                <Ionicons name="home" size={size} color={color} />
            ),
        }}/>
      <Tab.Screen
        name="Upload"
        component={UploadScreen}
        options={{
            tabBarIcon: ({color, size }) => (
                <Ionicons name="add-circle" size={size} color={color} />
            ),
        }}/>
    </Tab.Navigator>
  );
}