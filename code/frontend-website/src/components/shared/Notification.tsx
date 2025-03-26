import React, { useEffect } from "react";

const NotificationScheduler = () => {
  // Hardcoded time in "HH:MM" format, e.g., 2:00 PM as "14:00"
  const scheduledTime = "14:00";

  // Function to trigger the notification
  const triggerNotification = () => {
    if (!("Notification" in window)) {
      console.error("This browser does not support desktop notifications.");
      return;
    }
    new Notification("Time Alert", {
      body: "It's time for your scheduled activity!",
    });
  };

  useEffect(() => {
    // Request permission if not already granted
    if (
      Notification.permission !== "granted" &&
      Notification.permission !== "denied"
    ) {
      Notification.requestPermission();
    }

    // Calculate the delay until the scheduled time
    const calculateDelay = (timeStr: string) => {
      const [targetHour, targetMinute] = timeStr.split(":").map(Number);
      const now = new Date();
      const target = new Date();
      target.setHours(targetHour, targetMinute, 0, 0);

      // If the target time has already passed today, schedule for tomorrow
      if (target <= now) {
        target.setDate(target.getDate() + 1);
      }
      return target.getTime() - now.getTime();
    };

    const delay = calculateDelay(scheduledTime);

    // Set the timer for the notification
    const timerId = setTimeout(() => {
      triggerNotification();
    }, delay);

    // Clean up the timer if the component unmounts or if scheduledTime changes in the future
    return () => clearTimeout(timerId);
  }, [scheduledTime]);

  return (
    <div>
      <h1>Notification Scheduler</h1>
      <p>
        A notification is scheduled to trigger automatically at {scheduledTime}.
      </p>
    </div>
  );
};

export default NotificationScheduler;
