// mobile/components/SwipeableRow.js
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Animated } from 'react-native';
import { Swipeable } from 'react-native-gesture-handler';

const SwipeableRow = ({ item, onDelete, onNavigate }) => {
  // 定义右侧滑动出现的内容
  const renderRightActions = (progress, dragX) => {
    const trans = dragX.interpolate({
      inputRange: [-80, 0],
      outputRange: [0, 80],
      extrapolate: 'clamp',
    });
    return (
      <TouchableOpacity onPress={onDelete} style={styles.deleteButton}>
        <Animated.Text style={[styles.deleteButtonText, { transform: [{ translateX: trans }] }]}>
          删除
        </Animated.Text>
      </TouchableOpacity>
    );
  };

  return (
    <Swipeable renderRightActions={renderRightActions}>
      <TouchableOpacity style={styles.sessionItem} onPress={onNavigate}>
        <Text style={styles.sessionTitle} numberOfLines={1}>{item.title}</Text>
        <Text style={styles.sessionPreview} numberOfLines={1}>{item.preview}</Text>
      </TouchableOpacity>
    </Swipeable>
  );
};

const styles = StyleSheet.create({
  deleteButton: {
    backgroundColor: 'red',
    justifyContent: 'center',
    alignItems: 'flex-end',
    width: 80,
  },
  deleteButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
    padding: 20,
  },
  sessionItem: { 
    backgroundColor: 'white', 
    padding: 20, 
    borderBottomWidth: 1, 
    borderBottomColor: '#eee' 
  },
  sessionTitle: { 
    fontSize: 16, 
    fontWeight: 'bold', 
    marginBottom: 5 
  },
  sessionPreview: { 
    fontSize: 14, 
    color: '#666' 
  },
});

export default SwipeableRow;