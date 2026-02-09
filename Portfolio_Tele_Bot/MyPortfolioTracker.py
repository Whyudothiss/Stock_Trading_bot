import os
import logging
from datetime import datetime, timedelta
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from pymongo import MongoClient
from bson import ObjectId
import asyncio
from dotenv import load_dotenv
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

load_dotenv()

# Logging set Up
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI")
BOT_Token = os.getenv("BOT_TOKEN")

if not MONGO_URI or not BOT_Token:
    raise ValueError("Please set MONGO_URI and BOT_TOKEN environment variables")

# MongoDB setup
try:
    from urllib.parse import quote_plus
    # If URI contains username and password, encode them
    if '@' in MONGO_URI:
        # Split the URI into parts
        prefix = MONGO_URI.split('://', 1)[0]
        rest = MONGO_URI.split('://', 1)[1]
        credentials = rest.split('@')[0]
        host_part = rest.split('@')[1]
        username = credentials.split(':')[0]
        password = credentials.split(':')[1]
        
        # URL encode the username and password
        encoded_username = quote_plus(username)
        encoded_password = quote_plus(password)
        
        # Reconstruct the URI
        MONGO_URI = f"{prefix}://{encoded_username}:{encoded_password}@{host_part}"

    # Connect with SSL settings for Atlas
    client = MongoClient(
        MONGO_URI,
        tlsAllowInvalidCertificates=True  # This bypasses certificate verification
    )
    # Test the connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")
    raise
db = client.portfoloio_tracker

# Collections
users_collection = db.users
transactions_collection = db.transactions
portfolio_values_collection = db.portfolio_values

(TICKER, QUANTITY, PRICE, TRIM_TYPE, TRIM_VALUE, 
 DELETE_TRANSACTION, DELETE_CONFIRM, EDIT_SELECT, EDIT_FIELD, EDIT_VALUE) = range(10)

class PortfolioTracker():
    def __init__(self):
        self.user_data = {}
        
    def get_real_time_price(self, ticker):
        """Get real-time stock price from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.info.get('regularMarketPrice')
            return current_price if current_price else None
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {str(e)}")
            return None

    def calculate_portfolio_value(self, user_id):
        """Calculate current portfolio value using real-time prices"""
        total_value = 0
        holdings = {}
        
        # Get all buy transactions
        transactions = list(transactions_collection.find({"user_id": user_id}).sort("timestamp", 1))
        
        for tx in transactions:
            ticker = tx['ticker']
            quantity = tx['quantity']
            if tx['action'] == 'buy':
                holdings[ticker] = holdings.get(ticker, 0) + quantity
            else:  # sell or trim
                holdings[ticker] = holdings.get(ticker, 0) - quantity
                
        # Calculate current value using real-time prices
        for ticker, shares in holdings.items():
            if shares > 0:
                current_price = self.get_real_time_price(ticker)
                if current_price:
                    total_value += shares * current_price
                    
        return total_value

    async def update_portfolio_value(self, user_id):
        """Update daily portfolio value in database"""
        current_value = self.calculate_portfolio_value(user_id)
        portfolio_values_collection.insert_one({
            "user_id": user_id,
            "value": current_value,
            "timestamp": datetime.now()
        })

    def generate_equity_curve(self, user_id):
        """Generate equity curve plot"""
        # Get historical portfolio values
        values = list(portfolio_values_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", 1))
        
        if not values:
            return None
            
        # Create dataframe
        df = pd.DataFrame(values)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['value'])
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Save plot to bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        username = update.effective_user.username or update.effective_user.first_name

        # Ensure user is in the data dictionary
        if not users_collection.find_one({"user_id": user_id}):
            users_collection.insert_one({
                "user_id": user_id,
                "username": username,
                "create_at": datetime.now()
            })

        welcome_msg = """ 🎯 Welcome to myPortfolioTracker! 🎯

Your personal stock portfolio management bot. Here are the available commands:

📊 Portfolio Commands:
/buy - Record a stock purchase
/sell - Record a stock sale  
/trim - Partial sell (by % or quantity)
/current - View current holdings with P&L
/past - View sold positions with P&L
/pnl - Overall profit/loss summary
/value - Real-time portfolio value
/equity - Display equity curve

🛠 Management Commands:
/delete - Delete a transaction
/exit - Cancel current operation
/help - Show this help message

Ready to track your investments! Use /buy to record your first purchase."""


        await update.message.reply_text(welcome_msg)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.start(update, context)

    async def equity_curve(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Display the equity curve"""
        user_id = update.effective_user.id
        
        # Update current portfolio value
        await self.update_portfolio_value(user_id)
        
        # Generate equity curve
        plot_buffer = self.generate_equity_curve(user_id)
        
        if plot_buffer:
            current_value = self.calculate_portfolio_value(user_id)
            caption = f"📈 Your Portfolio Equity Curve\nCurrent Value: ${current_value:,.2f}"
            await update.message.reply_photo(plot_buffer, caption=caption)
        else:
            await update.message.reply_text("No portfolio data available yet. Start trading to see your equity curve!")

    async def value(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Display current portfolio value with real-time prices"""
        user_id = update.effective_user.id
        current_value = self.calculate_portfolio_value(user_id)
        
        # Get holdings with current prices
        holdings = {}
        transactions = list(transactions_collection.find({"user_id": user_id}).sort("timestamp", 1))
        
        for tx in transactions:
            ticker = tx['ticker']
            quantity = tx['quantity']
            if tx['action'] == 'buy':
                holdings[ticker] = holdings.get(ticker, 0) + quantity
            else:  # sell or trim
                holdings[ticker] = holdings.get(ticker, 0) - quantity
        
        message = "📊 Current Portfolio Value (Real-time)\n\n"
        
        for ticker, shares in holdings.items():
            if shares > 0:
                current_price = self.get_real_time_price(ticker)
                if current_price:
                    position_value = shares * current_price
                    message += f"{ticker}: {shares} shares @ ${current_price:.2f} = ${position_value:,.2f}\n"
        
        message += f"\n💰 Total Portfolio Value: ${current_value:,.2f}"
        await update.message.reply_text(message)


    # Buy transaction flow
    async def buy_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start buy trasnsaction"""
        context.user_data['action'] = 'buy'
        await update.message.reply_text("Let's record a stock purchase!\n\n Whats the ticker symbol (e.g., AAPL):")
        return TICKER
    
    async def get_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Get ticker symbol"""
        ticker = update.message.text.strip().upper()
        context.user_data['ticker'] = ticker

        await update.message.reply_text(f"Stock: {ticker}\n\n How many shares did you buy?")
        return QUANTITY
    
    async def get_quantity(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            quantity = float(update.message.text)

            if quantity <= 0:
                await update.message.reply_text("Please enter a positive number")
                return QUANTITY

            context.user_data['quantity'] = quantity
            ticker = context.user_data['ticker']
            action = context.user_data['action']
            
            # For sell/trim, validate quantity immediately
            if action in ['sell', 'trim']:
                user_id = update.effective_user.id
                current_position = self.get_current_positions(user_id, ticker)
                if current_position < quantity:
                    await update.message.reply_text(f"❌ Error: You only own {current_position} shares of {ticker}, "
                        f"but trying to {action} {quantity} shares.\n\nPlease enter a valid quantity.")
                    return QUANTITY

            # Dynamic message based on action
            action_text = "buy" if action == "buy" else "sell"
            await update.message.reply_text(f"Stock: {ticker}\nQuantity: {quantity} shares\n\nWhat price did you {action_text} at? (USD)")
            return PRICE
        
        except ValueError:
            await update.message.reply_text("Please enter a valid number")
            return QUANTITY
            
    async def get_price_and_execute(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            price = float(update.message.text)    
            if price <= 0:
                await update.message.reply_text("PLease enter a positive price")
                return PRICE

            user_id = update.effective_user.id
            ticker = context.user_data['ticker']
            quantity = context.user_data['quantity']
            action = context.user_data['action']
            total_value = quantity * price

            transaction ={
                "user_id": user_id,
                "ticker": ticker,
                "action": action,
                "price": price,
                "quantity": quantity,
                "total_value": total_value,
                "timestamp": datetime.now()
            }

            transactions_collection.insert_one(transaction)

            action_emoji = "📈" if action == "buy" else "📉"
            confirmation = f"""
{action_emoji} Transaction Recorded!
Stock: {ticker}
Action: {action.upper()}
Quantity: {quantity} shares
Price: ${price:.2f}
Total: ${total_value:.2f}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

✅ Successfully added to your portfolio! 
           """
            await update.message.reply_text(confirmation)
            return ConversationHandler.END
        except ValueError:
            await update.message.reply_text("Please enter a valid price")
            return PRICE

    async def sell_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start sell transaction"""
        user_id = update.effective_user.id
        holdings = self.get_user_holdings(user_id)
        
        if not holdings:
            await update.message.reply_text("📉 You don't have any holdings to sell.")
            return ConversationHandler.END
        
        context.user_data['action'] = 'sell'
        
        # Show available tickers
        message = "📉 Let's record a stock sale!\n\n💼 Your current holdings:\n\n"
        for ticker, data in holdings.items():
            message += f"• {ticker}: {data['quantity']:.2f} shares\n"
        message += "\nWhat ticker would you like to sell?"
        
        await update.message.reply_text(message)
        return TICKER

    async def trim_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start trim transaction"""
        user_id = update.effective_user.id
        holdings = self.get_user_holdings(user_id)
        
        if not holdings:
            await update.message.reply_text("✂️ You don't have any holdings to trim.")
            return ConversationHandler.END
        
        # Show available tickers
        message = "✂️ Let's record a stock trim (partial sell)!\n\n💼 Your current holdings:\n\n"
        for ticker, data in holdings.items():
            message += f"• {ticker}: {data['quantity']:.2f} shares\n"
        message += "\nWhat ticker would you like to trim?"
        
        await update.message.reply_text(message)
        return TICKER

    async def get_trim_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Get ticker symbol for trim"""
        ticker = update.message.text.strip().upper()
        user_id = update.effective_user.id

        current_position = self.get_current_positions(user_id, ticker)
        if current_position <= 0:
            await update.message.reply_text(f"❌ You don't currently own any {ticker} shares.")
            return ConversationHandler.END
        
        context.user_data['ticker'] = ticker
        context.user_data['current_position'] = current_position

        keyboard = [['Percentage', 'Quantity']]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)

        await update.message.reply_text(
            f"Stock: {ticker}\nCurrent position: {current_position} shares\n\n"
            "How would you like to trim?",
            reply_markup=reply_markup
        )
        return TRIM_TYPE
    
    async def get_trim_type(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        trim_type = update.message.text.strip().lower()
        if trim_type not in ['percentage', 'quantity']:
            await update.message.reply_text("Please choose either Percentage or Quantity")
            return TRIM_TYPE
        ticker = context.user_data['ticker']
        context.user_data['trim_type'] = trim_type
        current_position = context.user_data['current_position']

        if trim_type == 'percentage':
            await update.message.reply_text(
                f"Stock: {ticker}\nCurrent: {current_position} shares\n\n"
                "What percentage would you like to sell? (e.g., 25 for 25%)",
                reply_markup=ReplyKeyboardRemove()
            )
        else:
            await update.message.reply_text(
                f"Stock: {ticker}\nCurrent: {current_position} shares\n\n"
                "How many shares would you like to sell?",
                reply_markup=ReplyKeyboardRemove()
            )
        return TRIM_VALUE

    async def get_trim_value(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            value = float(update.message.text)
            trim_type = context.user_data['trim_type']
            ticker = context.user_data['ticker']
            current_position = context.user_data['current_position']

            # For percentage trim type
            if trim_type == 'percentage':
                if value <= 0 or value > 100:
                    await update.message.reply_text("Please enter a percentage between 0 and 100")
                    return TRIM_VALUE
                quantity = (value / 100) * current_position
            # For quantity trim type
            else:
                if value <= 0 or value > current_position:
                    await update.message.reply_text(f"Please enter a quantity between 0 and {current_position}")
                    return TRIM_VALUE
                quantity = value
            
            context.user_data['quantity'] = quantity
            context.user_data['action'] = 'trim'

            await update.message.reply_text(f"Stock: {ticker}\n"
                f"Selling: {quantity:.2f} shares ({(quantity/current_position)*100:.1f}%)\n"
                f"Remaining: {current_position - quantity:.2f} shares\n\n"
                "At what price are you selling? (USD)"
            )
            return PRICE
        except ValueError:
            await update.message.reply_text("Please enter a valid number")
            return TRIM_VALUE

    def get_current_positions(self, user_id, ticker):
        """Get current positions for a user and ticker"""
        pipeline = [
            {"$match": {"user_id": user_id, "ticker": ticker}},
            {"$group": {
                "_id": "$ticker",
                "total_bought": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$quantity", 0]}},
                "total_sold": {"$sum": {"$cond": [{"$ne": ["$action", "buy"]}, "$quantity", 0]}}
            }}
        ]

        result = list(transactions_collection.aggregate(pipeline))
        if result:
            return result[0]["total_bought"] - result[0]["total_sold"]
        return 0
    
    def get_user_holdings(self, user_id):
        """Get all current holdings for a user"""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$ticker",
                "total_bought_qty": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$quantity", 0]}},
                "total_sold_qty": {"$sum": {"$cond": [{"$ne": ["$action", "buy"]}, "$quantity", 0]}},
                "total_bought_value": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$total_value", 0]}},
                "total_sold_value": {"$sum": {"$cond": [{"$ne": ["$action", "buy"]}, "$total_value", 0]}}
            }},
            {"$addFields": {
                "current_quantity": {"$subtract": ["$total_bought_qty", "$total_sold_qty"]},
                "current_cost": {"$subtract": [
                    "$total_bought_value",
                    {"$multiply": [{"$divide": ["$total_bought_value", "$total_bought_qty"]}, "$total_sold_qty"]}
                ]}
            }},
            {"$match": {"current_quantity": {"$gt": 0}}}
        ]

        results = list(transactions_collection.aggregate(pipeline))

        holdings = {}
        for result in results:
            ticker = result["_id"]
            # calculate weighted average cost for remaining shares
            avg_buy_price = result["total_bought_value"] / result["total_bought_qty"] if result["total_bought_qty"] > 0 else 0
            remaing_cost = avg_buy_price * result["current_quantity"]

            holdings[ticker] = {
                "quantity": result["current_quantity"],
                "total_cost": remaing_cost,
                "avg_cost": avg_buy_price
            }
        
        return holdings
    
    def get_past_positions_with_pnl(self, user_id):
        """Get past positions that were completely sold with P&L data (excluding reopened positions)"""
        # First, get current holdings to exclude reopened positions
        current_holdings = self.get_user_holdings(user_id)
        current_tickers = set(current_holdings.keys())
        
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$ticker",
                "total_bought_qty": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$quantity", 0]}},
                "total_sold_qty": {"$sum": {"$cond": [{"$ne": ["$action", "buy"]}, "$quantity", 0]}},
                "total_bought_value": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$total_value", 0]}},
                "total_sold_value": {"$sum": {"$cond": [{"$ne": ["$action", "buy"]}, "$total_value", 0]}}
            }},
            {"$match": {
                "total_sold_qty": {"$gt": 0},
                "$expr": {"$gte": ["$total_sold_qty", "$total_bought_qty"]}  # Sold >= bought (completely closed)
            }}
        ]
        
        results = list(transactions_collection.aggregate(pipeline))

        past_positions = {}

        for result in results:
            ticker = result["_id"]
            
            # Skip if this ticker is currently held (reopened position)
            if ticker in current_tickers:
                continue
                
            bought_qty = result["total_bought_qty"]
            sold_qty = min(result["total_sold_qty"], bought_qty)

            past_positions[ticker] = {
                "total_bought": result["total_bought_value"],
                "total_sold": (result["total_sold_value"] / result["total_sold_qty"]) * sold_qty if result["total_sold_qty"] > 0 else 0,
                "avg_buy_price": result["total_bought_value"] / bought_qty if bought_qty > 0 else 0,
                "avg_sell_price": result["total_sold_value"] / result["total_sold_qty"] if result["total_sold_qty"] > 0 else 0,
            }
        return past_positions
    
    async def current_holdings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current holdings with real-time prices and unrealized P&L"""
        user_id = update.effective_user.id
        holdings = self.get_user_holdings(user_id)

        if not holdings:
            await update.message.reply_text("📊 You don't have any current holdings.")
            return
        
        message = "📊 Your Current Holdings:\n\n"
        total_invested = 0
        total_current_value = 0
        
        for ticker, data in holdings.items():
            avg_price = data["total_cost"] / data["quantity"]
            total_cost = data["total_cost"]
            total_invested += total_cost
            
            # Get current price
            current_price = self.get_real_time_price(ticker)
            
            message += f"🏷 {ticker}\n"
            message += f"   Shares: {data['quantity']:.2f}\n"
            message += f"   Avg Price: ${avg_price:.2f}\n"
            message += f"   Total Cost: ${total_cost:.2f}\n"
            
            if current_price:
                current_value = current_price * data['quantity']
                unrealized_pnl = current_value - total_cost
                unrealized_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
                total_current_value += current_value
                
                pnl_emoji = "📈" if unrealized_pnl >= 0 else "📉"
                message += f"   Current Price: ${current_price:.2f}\n"
                message += f"   {pnl_emoji} Unrealized P&L: ${unrealized_pnl:.2f} ({unrealized_pct:+.1f}%)\n\n"
            else:
                message += f"   ⚠️ Unable to fetch current price\n\n"
        
        message += f"💰 Total Invested: ${total_invested:.2f}\n"
        
        if total_current_value > 0:
            total_unrealized = total_current_value - total_invested
            total_unrealized_pct = (total_unrealized / total_invested * 100) if total_invested > 0 else 0
            pnl_emoji = "📈" if total_unrealized >= 0 else "📉"
            message += f"💵 Current Value: ${total_current_value:.2f}\n"
            message += f"{pnl_emoji} Total Unrealized P&L: ${total_unrealized:.2f} ({total_unrealized_pct:+.1f}%)"

        await update.message.reply_text(message)

    async def past_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):   
        """Show past positions with P&L"""
        user_id = update.effective_user.id
        past_positions = self.get_past_positions_with_pnl(user_id)
        
        if not past_positions:
            await update.message.reply_text("📜 No past positions found.\n\nYou haven't sold any complete positions yet.")
            return
        
        message = "📜 Your Past Positions:\n\n"
        total_pnl = 0

        for ticker, data in past_positions.items():
            pnl = data["total_sold"] - data["total_bought"]
            pnl_pct = (pnl / data["total_bought"] * 100) if data["total_bought"] > 0 else 0
            total_pnl += pnl

            pnl_emoji = "📈" if pnl >= 0 else "📉"

            message += f"🏷 {ticker} {pnl_emoji}\n"
            message += f"   Avg Buy: ${data['avg_buy_price']:.2f}\n"
            message += f"   Avg Sell: ${data['avg_sell_price']:.2f}\n"
            message += f"   P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)\n\n"

        pnl_emoji = "📈" if pnl >= 0 else "📉"    
        message += f"{pnl_emoji} Total Realized P&L: ${total_pnl:.2f}"

        await update.message.reply_text(message)

    async def pnl_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show PnL summary"""
        user_id = update.effective_user.id
        past_positions = self.get_past_positions_with_pnl(user_id)
        current_holdings = self.get_user_holdings(user_id)

        # Realised P&L from past positions
        realized_pnl = sum(
            (data["total_sold"] - data["total_bought"])
            for data in past_positions.values()
        )

        # Current investments and unrealized P&L
        total_invested = sum(
            data["total_cost"] for data in current_holdings.values()
        )
        
        # Calculate unrealized P&L
        total_current_value = 0
        for ticker, data in current_holdings.items():
            current_price = self.get_real_time_price(ticker)
            if current_price:
                total_current_value += current_price * data['quantity']
        
        unrealized_pnl = total_current_value - total_invested if total_current_value > 0 else 0
        total_pnl = realized_pnl + unrealized_pnl

        message = "💰 P&L SUMMARY 💰\n\n"
        message += f"📈 Realized P&L: ${realized_pnl:.2f}\n"
        message += f"💼 Current Invested: ${total_invested:.2f}\n"
        
        if total_current_value > 0:
            message += f"💵 Current Value: ${total_current_value:.2f}\n"
            unrealized_emoji = "📈" if unrealized_pnl >= 0 else "📉"
            message += f"{unrealized_emoji} Unrealized P&L: ${unrealized_pnl:.2f}\n\n"
            total_emoji = "🎉" if total_pnl >= 0 else "📊"
            message += f"{total_emoji} TOTAL P&L: ${total_pnl:.2f}\n\n"
        else:
            message += "\n"
        
        message += f"📊 Positions: {len(current_holdings)} open, {len(past_positions)} closed\n\n"

        if total_pnl >= 0:
            message += "🎉 Great job! You're in profit territory!"
        else:
            message += "📊 Keep tracking! Every trade is a learning opportunity."

        await update.message.reply_text(message)


    async def delete_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start delete transaction flow"""
        user_id = update.effective_user.id
        transactions = list(transactions_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(20))
        
        if not transactions:
            await update.message.reply_text("🗑 No transactions found to delete.")
            return ConversationHandler.END
        
        # Store transactions in context for later reference
        context.user_data['transactions'] = transactions
        
        # Display recent transactions
        message = "🗑 DELETE TRANSACTION\n\n"
        message += "Recent transactions (last 20):\n\n"
        
        for i, tx in enumerate(transactions, 1):
            action_emoji = "📈" if tx['action'] == 'buy' else "📉"
            date_str = tx['timestamp'].strftime('%Y-%m-%d %H:%M')
            message += f"{i}. {action_emoji} {tx['action'].upper()} {tx['ticker']} - {tx['quantity']} shares @ ${tx['price']:.2f} ({date_str})\n"
        
        message += "\nEnter the number of the transaction to delete (or /exit to cancel):"
        await update.message.reply_text(message)
        return DELETE_TRANSACTION
    
    async def delete_execute(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Execute the deletion"""
        try:
            selection = int(update.message.text)
            transactions = context.user_data.get('transactions', [])
            
            if selection < 1 or selection > len(transactions):
                await update.message.reply_text(f"❌ Invalid selection. Please enter a number between 1 and {len(transactions)}.")
                return DELETE_TRANSACTION
            
            # Get the selected transaction
            tx = transactions[selection - 1]
            
            # Delete from database
            result = transactions_collection.delete_one({"_id": tx['_id']})
            
            if result.deleted_count > 0:
                action_emoji = "📈" if tx['action'] == 'buy' else "📉"
                await update.message.reply_text(
                    f"✅ Transaction deleted successfully!\n\n"
                    f"{action_emoji} {tx['action'].upper()} {tx['ticker']} - {tx['quantity']} shares @ ${tx['price']:.2f}"
                )
            else:
                await update.message.reply_text("❌ Failed to delete transaction.")
            
            return ConversationHandler.END
            
        except ValueError:
            await update.message.reply_text("❌ Please enter a valid number.")
            return DELETE_TRANSACTION

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Cancel the conversation"""
        await update.message.reply_text(
            "❌ Operation cancelled.",
            reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END

async def setup_bot_commands(application: Application):
    """Set up bot command menu"""
    commands = [
        BotCommand("start", "🎯 Start the bot"),
        BotCommand("help", "❓ Show help message"),
        BotCommand("buy", "📈 Record stock purchase"),
        BotCommand("sell", "📉 Record stock sale"),
        BotCommand("trim", "✂️ Partial sell position"),
        BotCommand("current", "📊 View current holdings"),
        BotCommand("past", "📜 View closed positions"),
        BotCommand("pnl", "💰 P&L summary"),
        BotCommand("value", "💵 Real-time portfolio value"),
        BotCommand("equity", "📈 Display equity curve"),
        BotCommand("delete", "🗑 Delete transaction"),
        BotCommand("exit", "❌ Cancel current operation"),
    ]
    await application.bot.set_my_commands(commands)

def main():
    tracker = PortfolioTracker()

    # Create application
    application = Application.builder().token(BOT_Token).build()

    # Buy conversation handler
    buy_handler = ConversationHandler(
        entry_points = [CommandHandler('buy', tracker.buy_start)],
        states = {
            TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_ticker)],
            QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_quantity)],
            PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_price_and_execute)],     
        },
        fallbacks = [CommandHandler('cancel', tracker.cancel), CommandHandler('exit', tracker.cancel)],
        allow_reentry=False,
    )

    # Sell conversation handler
    sell_handler = ConversationHandler(
        entry_points=[CommandHandler('sell', tracker.sell_start)],
        states={
            TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_ticker)],
            QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_quantity)],
            PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_price_and_execute)],
        },
        fallbacks=[CommandHandler('cancel', tracker.cancel), CommandHandler('exit', tracker.cancel)],
        allow_reentry=False,
    )

    # Trim conversation handler
    trim_handler = ConversationHandler(
        entry_points=[CommandHandler('trim', tracker.trim_start)],
        states={
            TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_trim_ticker)],
            TRIM_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_trim_type)],
            TRIM_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_trim_value)],
            PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_price_and_execute)],
        },
        fallbacks=[CommandHandler('cancel', tracker.cancel), CommandHandler('exit', tracker.cancel)],
        allow_reentry=False,
    )
    
    # Delete conversation handler
    delete_handler = ConversationHandler(
        entry_points=[CommandHandler('delete', tracker.delete_start)],
        states={
            DELETE_TRANSACTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.delete_execute)],
        },
        fallbacks=[CommandHandler('cancel', tracker.cancel), CommandHandler('exit', tracker.cancel)],
        allow_reentry=False,
    )

    # Add handlers
    application.add_handler(CommandHandler("start", tracker.start))
    application.add_handler(CommandHandler("help", tracker.help_command))
    application.add_handler(buy_handler)
    application.add_handler(sell_handler)
    application.add_handler(trim_handler)
    application.add_handler(delete_handler)
    application.add_handler(CommandHandler("current", tracker.current_holdings))
    application.add_handler(CommandHandler("past", tracker.past_positions))
    application.add_handler(CommandHandler("pnl", tracker.pnl_summary))
    application.add_handler(CommandHandler("value", tracker.value))
    application.add_handler(CommandHandler("equity", tracker.equity_curve))

    # Set up bot command menu
    asyncio.get_event_loop().run_until_complete(setup_bot_commands(application))

    # Configure port for cloud deployment (Railway or Render)
    port = int(os.getenv("PORT", 8080))
    print(f"🎯 myPortfolioTracker bot is starting on port {port}...")
    
    # Check if running on Railway or Render
    if os.getenv("RAILWAY_ENVIRONMENT"):
        print("Running on Railway - using webhooks")
        webhook_url = f"https://{os.getenv('RAILWAY_PUBLIC_DOMAIN')}"
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            webhook_url=webhook_url
        )
    elif os.getenv("RENDER"):
        print("Running on Render - using webhooks")
        # Render automatically provides RENDER_EXTERNAL_URL
        webhook_url = os.getenv("RENDER_EXTERNAL_URL", "")
        if webhook_url:
            application.run_webhook(
                listen="0.0.0.0",
                port=port,
                webhook_url=webhook_url
            )
        else:
            print("ERROR: RENDER_EXTERNAL_URL not found!")
            # Fallback to polling for testing
            application.run_polling(allowed_updates=Update.ALL_TYPES)
    else:
        print("Running locally - using polling")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    
if __name__ == '__main__':
    main()    