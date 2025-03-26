import { useEffect, useState } from "react";
import user from "../constant/user.json";

interface HeroOffer {
  contentData: any;
  title: string;
  desc: string;
}

const Personal = () => {
  const [heroOffer, setHeroOffer] = useState<HeroOffer | null>(null);
  const [bankOffers, setBankOffers] = useState<any[]>([]);
  const [merchantOffers, setMerchantOffers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://localhost:8000/bank-recommend", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(user),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("First fetch:", data);

        setHeroOffer(data[0]);

        return fetch("http://localhost:5000/content", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ offers: data[0], user: user }),
        });
      })
      .then((response) => response.json())
      .then((secondData) => {
        console.log("Second fetch:", secondData);

        setHeroOffer(secondData);
      })
      .catch((err) => {
        console.error(err);
      });
  }, []);
  //  bank offers
  useEffect(() => {
    fetch("http://localhost:8000/bank-recommend", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(user),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("First fetch bank:", data);
        // Create an array of promises for each item in the data array.
        const contentPromises = data.slice(0, 6).map((item: any) => {
          return fetch("http://localhost:5000/content", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ offers: item, user: user }),
          }).then((response) => response.json());
        });
        // Wait for all fetches to complete.
        return Promise.all(contentPromises);
      })
      .then((allContentData) => {
        console.log("All bank fetch:", allContentData);
        // allContentData is an array containing the responses for each item.
        setBankOffers(allContentData);
      })
      .catch((err) => {
        console.error(err);
      });
  }, []);

  //  merchant offers
  useEffect(() => {
    fetch("http://localhost:8000/merchant-recommend", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(user),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("First fetch bank:", data);
        // Create an array of promises for each item in the data array

        const contentPromises = data.slice(0, 6).map((item: any) => {
          console.log(
            "Json data",
            JSON.stringify({ offers: item, user: user })
          );
          return fetch("http://localhost:5000/content", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ offers: item, user: user }),
          }).then((response) => response.json());
        });
        // Wait for all fetches to complete.
        return Promise.all(contentPromises);
      })
      .then((allContentData) => {
        console.log("All merchant:", allContentData);
        // allContentData is an array containing the responses for each item.
        setMerchantOffers(allContentData);
      })
      .catch((err) => {
        console.error(err);
      })
      .finally(() => setLoading(false));
  }, []);

  const financialGuidanceData = [
    {
      image:
        "https://www17.wellsfargomedia.com/assets/images/contextual/responsive/smlpromo/wfi000_ph_g_557715963_616x353.jpg",
      title: "Get tools. Get tips. Get peace of mind.",
      description:
        "Discover digital tools to help you budget, save, manage credit, and more.",
      action: "Access the toolkit",
    },
    {
      image:
        "https://www17.wellsfargomedia.com/assets/images/contextual/responsive/smlpromo/wfi_ph_a_380700712-investingmoney_616x353.jpg",
      title: "Take the guess work out of investing",
      description:
        "Explore your options and see how you can start investing today.",
      action: "Get started",
    },
    {
      image:
        "https://www17.wellsfargomedia.com/assets/images/contextual/responsive/smlpromo/financial-goals_616x353.jpg",
      title: "Your Money. Your Goals. Your Future.",
      description:
        "Setting financial goals is a powerful first step you can take today.",
      action: "Get started",
    },
  ];

  if (loading) {
    return (
      <div className="flex items-center flex-col justify-center h-[83vh]">
        <div className="w-16 h-16 border-4 border-t-4 border-t-red-500 mb-4 border-gray-200 rounded-full animate-spin"></div>
        <p>Creating your personalised banking experience...</p>
      </div>
    );
  }

  return (
    <div className="mx-[15vw]  ">
      {/* hero section  */}
      <div className="bg-gradient-to-r px-[5%] flex flex-row pt-10 w-full h-[550px] from-red-700 via-red-600 to-red-700">
        <div
          id="login-card"
          className="w-[400px] h-[80%] flex flex-col bg-white rounded-xl shadow-xl"
        >
          <h1 className="text-xl font-semibold mb-1 m-5">Good morning</h1>
          <p className="text-[14px] mx-5">Sign on to manage your accounts.</p>
          <input className="border-b-1 mx-5 py-2 mt-4" placeholder="Username" />
          <input className="border-b-1 mx-5 py-2 mt-4" placeholder="Password" />
          <div className="flex flex-row mx-5 items-center text-[14px] justify-start gap-2 mt-6">
            <div className="w-4 h-4 border-1"></div> Save username
          </div>
          <div className="flex flex-row mx-5 items-center mt-4 justify-between">
            <button className="bg-[#d71e28] text-white w-[60%] py-2 rounded-full">
              Sign On
            </button>
            <button>Enroll</button>
          </div>
          <div className="bg-[#f4f0ed] mt-4 flex flex-col justify-center text-[14px] px-5 h-full w-full rounded-b-xl">
            <p>Forgot username or password?</p>
            <p>Security Center</p>
            <p>Privacy, Cookies, and Legal</p>
          </div>
        </div>

        {/* Render heroOffer once it's loaded */}
        {heroOffer && (
          <div className=" flex my-20 items-start  flex-col text-white ml-[15%]">
            <h1 className=" text-6xl mb-10 font-semibold">
              {heroOffer?.contentData?.title}
            </h1>
            <p className=" text-2lx mb-10 font-semibold">
              {heroOffer?.contentData?.desc}
            </p>
            <button className="bg-white border-1 text-black px-10 py-2 rounded-full">
              Get Started {">>"}
            </button>
          </div>
        )}
      </div>
      {/* bank offers  */}
      <div className="flex flex-col items-center mt-10">
        {bankOffers && (
          <div className="flex flex-row justify-between gap-5 mt-10">
            {bankOffers &&
              bankOffers?.slice(2, 6)?.map((offer) => (
                <div
                  key={offer?.title}
                  className="flex bg-[#f4f0ed] w-[15vw] min-h-[12vw] items-center  flex-col justify-between  mb-10 shadow-md rounded-lg p-5"
                >
                  <img
                    className="w-12 h-12"
                    src={
                      offer?.ProductType?.toLowerCase()?.includes("card")
                        ? "https://www17.wellsfargomedia.com/assets/images/contextual/responsive/smlprimary/creditcard_color_gradient_64x64x.png"
                        : "https://www17.wellsfargomedia.com/assets/images/contextual/responsive/smlprimary/wf_icon_piggybank_rgb_f1_gradient_64x64.png"
                    }
                  />
                  <div className="flex flex-col items-start ">
                    <h1 className="text-xl font-medium mb-5">
                      {offer?.contentData?.title}
                    </h1>
                    <p className="text-md font-normal ">
                      {offer?.contentData?.desc}
                    </p>
                  </div>
                  <p className=" text-purple-800 my-5 text-left w-full cursor-pointer ">
                    See offer details {">"}
                  </p>
                </div>
              ))}
          </div>
        )}
      </div>
      {/* first banner  */}
      <div className=" py-[5%] px-[5%] bg-cover bg-center bg-[url('https://www17.wellsfargomedia.com/assets/images/contextual/responsive/lpromo/choice_hplp_overlook_noflag_1600x700.jpg')] h-[600px] w-full">
        <h1 className="text-4xl font-light">Earn up to 60,000</h1>
        <h1 className="text-4xl my-5 font-light">bonus points</h1>
        <p className="text-xl mb-5 ">on qualifying purchases. Terms apply.</p>
        <button className="bg-white px-10 py-2 rounded-full border-1">
          {" "}
          Learn more
        </button>
      </div>
      {/* merchant offers  */}
      <div className="flex flex-col items-center mt-10">
        <div className="bg-[#ffcd41] w-20 h-0.5 "></div>
        <h1 className="text-4xl  mt-5">
          Exclusive Merchant Offers Curated for You
        </h1>
        <div className="flex flex-row justify-between gap-5 mt-10">
          {merchantOffers &&
            merchantOffers?.slice(0, 4).map((offer, index) => (
              <div
                key={offer?.title}
                className="flex bg-[#f4f0ed] w-[15vw] min-h-[12vh]  flex-col justify-between  mb-10 shadow-md rounded-lg p-5"
              >
                <div className="flex flex-col items-start ">
                  <h1 className="text-xl font-medium mb-5">
                    {offer?.contentData?.title}
                  </h1>
                  <p className="text-md font-normal ">
                    {offer?.contentData?.desc}
                  </p>
                </div>
                <p className=" text-purple-800 my-5 text-left cursor-pointer ">
                  See offer details {">"}
                </p>
              </div>
            ))}
        </div>
      </div>
      {/* banner 2  */}
      <div className="bg-[url('D:/projects/wells-hackathon/wellsfargo/src/assets/image.png')] h-[600px] w-full bg-cover bg-center"></div>
      {/* financial guidance section  */}
      <div className="flex flex-col items-center mt-10">
        <div className="bg-[#ffcd41] w-20 h-0.5 "></div>
        <h1 className="text-4xl  mt-5">Financial Guidance and Support</h1>
        <div className="flex flex-row justify-between mt-10">
          {financialGuidanceData.map((data, idx) => (
            <div
              key={data?.title}
              className="flex bg-[#fff] w-[30%]  flex-col justify-between  mb-10 shadow-md rounded-lg "
            >
              <img
                className={`w-full h-auto rounded-t-lg object-center`}
                src={data.image}
                alt=""
              />
              <div className=" m-5 flex flex-col items-start ">
                <h1 className="text-lg font-medium mb-5">{data.title}</h1>
                <p className="text-md font-normal ">{data.description}</p>
              </div>
              <button className=" m-5 text-center border-1 w-[60%] py-2 rounded-full cursor-pointer ">
                {data.action}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Personal;
