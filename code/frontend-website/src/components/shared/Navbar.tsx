import React from "react";
import logo from "../../assets/logo.png";
const Navbar = () => {
  const navLinks = ["ATMs/Locations", "Help", "About Us", "Español"];

  const secondNavLinks = [
    "Personal",
    "Small Business",
    "Commercial Banking",
    "Corporate & Investment Banking",
  ];
  const thirdNavLinks = [
    "Checking",
    "Savings & CDs",
    "Home Loans",
    "Credit Cards",
    "Personal Loans",
    "Auto Loans",
    "Premier",
    "Education & Tools",
  ];

  return (
    <nav>
      <div className="w-[100vw] bg-[#d71e28] h-[6vh] flex flex-row items-center px-[15vw] justify-between ">
        <img src={logo} />
        <div>
          {navLinks.map((link, index) => (
            <a key={index} href="#" className="text-white  ml-5">
              {link}
            </a>
          ))}
          <button></button>
          <button className="bg-white px-10 py-2 rounded-full ml-5 ">
            Sign On
          </button>
        </div>
      </div>
      <div className="w-[100vw] bg-[#ffcd41] h-[0.5vh] "></div>
      <div className="  w-[70vw] bg-[#f4f0ed] h-[5vh] flex flex-row items-center ml-[15vw]  justify-start ">
        {secondNavLinks.map((link, index) => (
          <a key={index} href="#" className=" ml-5">
            {link}
          </a>
        ))}
      </div>
      <div className="  w-[70vw]  h-[5vh] flex flex-row items-center ml-[15vw]  justify-start ">
        {thirdNavLinks.map((link, index) => (
          <a key={index} href="#" className=" ml-5">
            {link}
          </a>
        ))}
      </div>
    </nav>
  );
};

export default Navbar;
