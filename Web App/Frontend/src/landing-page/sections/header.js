import { Button } from "@mui/material";
import "./header.css";

function HeaderSection() {
  return (
    <div className="header">
      <div>
        <img src="icons/logo_trans.png" className="header-logo" />
      </div>
      <div className="nav-bar" style={{ color: "#fff" }}>
        <Button style={{ color: "#fff" }}>Home</Button>
        <Button style={{ color: "#fff" }}>About Us</Button>
        <Button style={{ color: "#fff" }}>Contact Us</Button>
      </div>
      <div>
        <Button
          variant="contained"
          style={{
            backgroundColor: "#fff",
            borderRadius: 20,
            color: "#1d2671",
          }}
        >
          Login
        </Button>
      </div>
    </div>
  );
}

export default HeaderSection;
