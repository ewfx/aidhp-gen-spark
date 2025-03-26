import { useEffect, useState } from "react";
import Footer from "./components/shared/Footer";
import Navbar from "./components/shared/Navbar";
import Personal from "./pages/Personal";
import "./App.css";
import data from "./constant/user.json";

function App() {
  const [showNotification, setShowNotification] = useState(false);

  interface LlmResponse {
    title: string;
    desc: string;
  }
  const [llmResponse, setLlmResponse] = useState<LlmResponse | null>(null);

  const triggerNotification = () => {
    setShowNotification(true);
  };
  const calculateDelay = (timeStr: string) => {
    const now = new Date();
    let target = new Date();
    if (timeStr.length === 5 && timeStr.includes(":")) {
      const [hours, minutes] = timeStr.split(":").map(Number);
      target.setHours(hours, minutes, 0, 0);
    } else {
      target = new Date(timeStr.replace(" ", "T"));
    }
    if (target <= now) {
      target.setDate(target.getDate() + 1);
    }
    return target.getTime() - now.getTime();
  };

  useEffect(() => {
    if (data.notification && data.notification.length > 0) {
      const now = new Date();
      console.log("Current time:", now.toString());
      let nextNotification = null;

      for (const notif of data.notification) {
        const notifActiveTime = notif["active-time"];
        let notifTime: Date;

        if (notifActiveTime.length === 5 && notifActiveTime.includes(":")) {
          const [hours, minutes] = notifActiveTime.split(":").map(Number);
          notifTime = new Date();
          notifTime.setHours(hours, minutes, 0, 0);
        } else {
          notifTime = new Date(notifActiveTime.replace(" ", "T"));
        }
        console.log("Notification time:", notifTime.toString());
        if (notifTime > now) {
          nextNotification = notif;
          break;
        }
      }

      if (!nextNotification) {
        console.log("No upcoming notifications found.");
        return;
      }
      const delay = calculateDelay(nextNotification["active-time"]);
      console.log("Calculated delay (ms):", delay);
      fetch("http://localhost:5000/notification", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(nextNotification),
      })
        .then((response) => response.json())
        .then((data) => {
          setLlmResponse(data);
          console.log("LLM Response:", data);
        })
        .catch((err) =>
          console.error("Error fetching notification data:", err)
        );
      const timerId = setTimeout(() => {
        triggerNotification();
      }, delay);
      return () => clearTimeout(timerId);
    }
  }, []);

  return (
    <>
      <Navbar />
      <Personal />
      <Footer />
      {showNotification && (
        <div className="notification-popup">
          <div className="notification-content">
            <h3>{llmResponse?.title}</h3>
            <p>{llmResponse?.desc}</p>
            <button onClick={() => setShowNotification(false)}>Close</button>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
